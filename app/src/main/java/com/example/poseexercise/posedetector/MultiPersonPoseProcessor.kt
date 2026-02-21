package com.example.poseexercise.posedetector

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Rect
import android.util.Log
import com.example.poseexercise.data.PostureResult
import com.example.poseexercise.data.TrackedPerson
import com.example.poseexercise.data.plan.Plan
import com.example.poseexercise.posedetector.classification.PoseClassifierProcessor
import com.example.poseexercise.util.VisionProcessorBase
import com.example.poseexercise.viewmodels.CameraXViewModel
import com.example.poseexercise.views.graphic.BlurOverlayGraphic
import com.example.poseexercise.views.graphic.GraphicOverlay
import com.example.poseexercise.views.graphic.PersonBoundingBoxGraphic
import com.example.poseexercise.views.graphic.PoseGraphic
import com.google.android.gms.tasks.Task
import com.google.android.gms.tasks.Tasks
import com.google.android.odml.image.MlImage
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.pose.Pose
import com.google.mlkit.vision.pose.PoseDetection
import com.google.mlkit.vision.pose.PoseDetector
import com.google.mlkit.vision.pose.PoseDetectorOptionsBase
import com.google.mlkit.vision.segmentation.Segmentation
import com.google.mlkit.vision.segmentation.SegmentationMask
import com.google.mlkit.vision.segmentation.Segmenter
import com.google.mlkit.vision.segmentation.selfie.SelfieSegmenterOptions
import java.nio.ByteBuffer
import java.util.concurrent.Executor
import java.util.concurrent.Executors
import kotlin.math.max
import kotlin.math.min

/**
 * Multi-person pose processor that orchestrates:
 * 1. Person detection via MediaPipe Object Detector (PersonDetectorProcessor)
 * 2. Pose detection + classification only for the selected person
 *
 * This replaces the single-person PoseDetectorProcessor when multi-person
 * support is needed. It extends VisionProcessorBase to integrate with
 * the existing CameraX pipeline.
 */
class MultiPersonPoseProcessor(
    private val context: Context,
    options: PoseDetectorOptionsBase,
    private val showInFrameLikelihood: Boolean,
    private val visualizeZ: Boolean,
    private val rescaleZForVisualization: Boolean,
    private val runClassification: Boolean,
    private val isStreamMode: Boolean,
    private var cameraXViewModel: CameraXViewModel? = null,
    notCompletedExercise: List<Plan>
) : VisionProcessorBase<MultiPersonPoseProcessor.MultiPersonResult>(context) {

    private val poseDetector: PoseDetector = PoseDetection.getClient(options)
    private val segmenterOptions = SelfieSegmenterOptions.Builder()
        .setDetectorMode(if (isStreamMode) SelfieSegmenterOptions.STREAM_MODE else SelfieSegmenterOptions.SINGLE_IMAGE_MODE)
        .build()
    private val segmenter: Segmenter = Segmentation.getClient(segmenterOptions)
    private val personDetector: PersonDetectorProcessor = PersonDetectorProcessor(context)
    private val classificationExecutor: Executor = Executors.newSingleThreadExecutor()

    private var poseClassifierProcessor: PoseClassifierProcessor? = null
    private var exercisesToDetect: List<String>? = null

    companion object {
        private const val TAG = "MultiPersonPoseProcessor"
        // Padding around bounding box when cropping for pose detection (as fraction of box size)
        private const val CROP_PADDING_RATIO = 0.15f
    }

    // Last-known bounding box for the selected person (used for persistent blur)
    private var lastSelectedBox: android.graphics.Rect? = null

    /**
     * Holds the combined results: detected persons + pose for the selected person.
     */
    inner class MultiPersonResult(
        val persons: List<TrackedPerson>,
        val selectedPose: Pose?,
        val selectedPersonBox: Rect?,
        classificationResult: Map<String, PostureResult>,
        val selectedMaskBitmap: Bitmap?
    ) {
        init {
            // Publish classification results to ViewModel
            if (classificationResult.isNotEmpty()) {
                cameraXViewModel?.postureLiveData?.postValue(classificationResult)
            }
        }
    }

    init {
        if (notCompletedExercise.isNotEmpty()) {
            exercisesToDetect = notCompletedExercise.map { it.exercise }
        }
    }

    override fun stop() {
        super.stop()
        poseDetector.close()
        segmenter.close()
        personDetector.close()
        cameraXViewModel = null
    }

    override fun detectInImage(image: InputImage): Task<MultiPersonResult> {
        // Get the latest camera bitmap for person detection
        val bitmap = latestBitmap ?: return Tasks.forResult(
            MultiPersonResult(emptyList(), null, null, emptyMap(), null)
        )

        return detectPersonsAndPose(bitmap, image)
    }

    override fun detectInImage(image: MlImage): Task<MultiPersonResult> {
        val bitmap = latestBitmap ?: return Tasks.forResult(
            MultiPersonResult(emptyList(), null, null, emptyMap(), null)
        )

        // Convert MlImage to InputImage is handled by the base; we use the bitmap path
        val inputImage = InputImage.fromBitmap(bitmap, 0)
        return detectPersonsAndPose(bitmap, inputImage)
    }

    /**
     * Core detection pipeline:
     * 1. Detect persons in the frame
     * 2. If a person is selected, crop and run pose detection on them
     * 3. Classify the pose for rep counting
     */
    private fun detectPersonsAndPose(
        bitmap: Bitmap,
        fullFrameImage: InputImage
    ): Task<MultiPersonResult> {

        // Step 1: Detect all persons
        val persons = try {
            personDetector.detectPersons(bitmap)
        } catch (e: Exception) {
            Log.e(TAG, "Person detection failed", e)
            emptyList()
        }

        // Mark the selected person
        val selectedId = cameraXViewModel?.selectedPersonId?.value ?: -1

        // Tell the person detector which ID is selected so it's protected from expiry
        personDetector.setSelectedId(selectedId)

        val annotatedPersons = persons.map { p ->
            if (p.trackingId == selectedId) p.copy(isSelected = true) else p
        }

        // Publish detected persons to ViewModel
        cameraXViewModel?.detectedPersonsLiveData?.postValue(annotatedPersons)

        // Auto-select if only one person detected and none selected
        if (selectedId == -1 && annotatedPersons.size == 1) {
            cameraXViewModel?.selectedPersonId?.postValue(annotatedPersons[0].trackingId)
        }

        // Step 2: Run pose detection on selected person only
        val selectedPerson = annotatedPersons.find { it.trackingId == selectedId }

        // Update or use last-known bounding box for persistent blur
        if (selectedPerson != null) {
            lastSelectedBox = selectedPerson.boundingBox
        }
        // If selected person is missing, lastSelectedBox retains the previous value

        if (selectedPerson == null || !runClassification) {
            // No selected person or classification not triggered yet
            return Tasks.forResult(
                MultiPersonResult(annotatedPersons, null, null, emptyMap(), null)
            )
        }

        // Crop the selected person's region from the bitmap with some padding
        val cropRect = computePaddedCropRect(selectedPerson.boundingBox, bitmap.width, bitmap.height)
        val croppedBitmap = try {
            Bitmap.createBitmap(
                bitmap,
                cropRect.left, cropRect.top,
                cropRect.width(), cropRect.height()
            )
        } catch (e: Exception) {
            Log.e(TAG, "Failed to crop bitmap for pose detection", e)
            return Tasks.forResult(
                MultiPersonResult(annotatedPersons, null, null, emptyMap(), null)
            )
        }

        // Run pose detection and segmentation concurrently on the cropped image
        val croppedInput = InputImage.fromBitmap(croppedBitmap, 0)
        val poseTask = poseDetector.process(croppedInput)
        val segmentTask = segmenter.process(croppedInput)

        return Tasks.whenAllComplete(poseTask, segmentTask)
            .continueWith(classificationExecutor) { task ->
                val pose = if (poseTask.isSuccessful) poseTask.result else null
                val mask = if (segmentTask.isSuccessful) segmentTask.result else null
                val maskBitmap = mask?.let { createMaskBitmap(it) }

                var classificationResult: Map<String, PostureResult> = HashMap()
                if (runClassification && pose != null && pose.allPoseLandmarks.isNotEmpty()) {
                    if (poseClassifierProcessor == null) {
                        poseClassifierProcessor = PoseClassifierProcessor(
                            context,
                            isStreamMode,
                            exercisesToDetect
                        )
                    }
                    classificationResult = poseClassifierProcessor!!.getPoseResult(pose)
                }

                // Recycle cropped bitmap
                if (!croppedBitmap.isRecycled) {
                    croppedBitmap.recycle()
                }

                MultiPersonResult(annotatedPersons, pose, cropRect, classificationResult, maskBitmap)
            }
    }

    override fun onSuccess(
        results: MultiPersonResult,
        graphicOverlay: GraphicOverlay
    ) {
        val selectedId = cameraXViewModel?.selectedPersonId?.value ?: -1

        // Portrait-mode blur: blur everything except the selected person
        if (selectedId >= 0) {
            // Use current bounding box if person is visible, otherwise use last-known position
            val selectedPerson = results.persons.find { it.trackingId == selectedId }
            val blurBox = selectedPerson?.boundingBox ?: lastSelectedBox

            if (blurBox != null) {
                val cameraBitmap = latestBitmap
                if (cameraBitmap != null && !cameraBitmap.isRecycled) {
                    graphicOverlay.add(
                        BlurOverlayGraphic(
                            graphicOverlay,
                            blurBox,
                            cameraBitmap,
                            results.selectedMaskBitmap,
                            results.selectedPersonBox
                        )
                    )
                }
            }
        }

        // Draw bounding boxes for all detected persons
        for (person in results.persons) {
            graphicOverlay.add(
                PersonBoundingBoxGraphic(
                    graphicOverlay,
                    person.boundingBox,
                    person.trackingId,
                    person.label,
                    person.isSelected,
                    person.confidence
                )
            )
        }

        // Draw pose skeleton for the selected person (if available and enabled)
        val pose = results.selectedPose
        val cropRect = results.selectedPersonBox
        val showSkeleton = cameraXViewModel?.showSkeletonLiveData?.value != false
        
        if (pose != null && cropRect != null && showSkeleton) {
            graphicOverlay.add(
                PoseGraphic(
                    graphicOverlay,
                    pose,
                    showInFrameLikelihood,
                    visualizeZ,
                    rescaleZForVisualization,
                    offsetX = cropRect.left.toFloat(),
                    offsetY = cropRect.top.toFloat()
                )
            )
        }
    }

    override fun onFailure(e: Exception) {
        Log.e(TAG, "Multi-person pose detection failed!", e)
    }

    override fun isMlImageEnabled(context: Context?): Boolean {
        return true
    }

    /**
     * Compute a padded crop rect that stays within image bounds.
     */
    private fun computePaddedCropRect(box: Rect, imageWidth: Int, imageHeight: Int): Rect {
        val padX = (box.width() * CROP_PADDING_RATIO).toInt()
        val padY = (box.height() * CROP_PADDING_RATIO).toInt()

        return Rect(
            max(0, box.left - padX),
            max(0, box.top - padY),
            min(imageWidth, box.right + padX),
            min(imageHeight, box.bottom + padY)
        )
    }

    /**
     * Converts a SegmentationMLMask into a Bitmap for fast drawing on canvas.
     */
    private fun createMaskBitmap(mask: SegmentationMask): Bitmap {
        val width = mask.width
        val height = mask.height
        val buffer = mask.buffer
        buffer.rewind()
        val pixels = IntArray(width * height)
        for (i in 0 until width * height) {
            val confidence = buffer.float
            val alpha = (confidence * 255).toInt().coerceIn(0, 255)
            pixels[i] = android.graphics.Color.argb(alpha, 0, 0, 0)
        }
        return Bitmap.createBitmap(pixels, 0, width, width, height, Bitmap.Config.ARGB_8888)
    }
}
