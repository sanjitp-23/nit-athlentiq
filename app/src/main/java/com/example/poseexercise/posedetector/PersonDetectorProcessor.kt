package com.example.poseexercise.posedetector

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Rect
import android.util.Log
import com.example.poseexercise.data.TrackedPerson
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.objectdetector.ObjectDetector
import com.google.mediapipe.tasks.vision.objectdetector.ObjectDetectorResult
import kotlin.math.abs
import kotlin.math.sqrt

/**
 * Detects and tracks people in camera frames using MediaPipe Object Detector
 * with a COCO-trained EfficientDet-Lite0 model.
 *
 * Uses a hybrid tracking approach with:
 * 1. EMA-smoothed centroid positions for stability during exercises
 * 2. Horizontal position as primary signal (people stay in same X during floor exercises)
 * 3. IoU as secondary signal when boxes overlap
 * 4. Size similarity to prevent swaps between people of different sizes
 */
class PersonDetectorProcessor(context: Context) {

    private val objectDetector: ObjectDetector
    private var nextTrackingId: Int = 0

    // Tracked state per person: smoothed centroid + last known box
    private val trackedStates = mutableMapOf<Int, TrackedState>()

    companion object {
        private const val TAG = "PersonDetectorProcessor"
        private const val MODEL_PATH = "efficientdet_lite0.tflite"
        private const val MAX_RESULTS = 3
        private const val SCORE_THRESHOLD = 0.35f
        private const val PERSON_CATEGORY = "person"

        // EMA smoothing factor for centroid positions (0 = no smoothing, 1 = instant)
        private const val CENTROID_EMA_ALPHA = 0.4f

        // Tracking weights (optimized for occlusion and crossing paths)
        private const val HORIZONTAL_WEIGHT = 0.35f   // Reduced to allow subjects to cross without instant ID swap
        private const val VERTICAL_WEIGHT = 0.15f     // Y-position less reliable during exercises
        private const val IOU_WEIGHT = 0.25f          // Increased to strongly prefer physical overlap 
        private const val SIZE_WEIGHT = 0.25f         // Increased to strongly prefer consistent subject size

        // Thresholds
        private const val MIN_MATCH_SCORE = 0.35f
        private const val MAX_HORIZONTAL_DIST_RATIO = 0.35f // Max horizontal dist as fraction of image width
        private const val MAX_VERTICAL_DIST_RATIO = 0.50f   // More lenient vertically for exercises

        // How many frames a person can be missing before we remove them
        private const val MAX_MISSING_FRAMES = 45  // ~1.5 seconds at 30fps
        private const val SELECTED_MAX_MISSING_FRAMES = 300  // ~10 seconds for selected person
    }

    // The currently selected person's tracking ID (protected from expiry)
    private var selectedTrackingId: Int = -1

    /**
     * Set the selected person's tracking ID so their state is preserved longer.
     */
    fun setSelectedId(id: Int) {
        selectedTrackingId = id
        // Mark the selected state in tracked states
        trackedStates.values.forEach { state ->
            state.isSelected = (state.trackingId == id)
        }
    }

    /**
     * Internal state for a tracked person, including smoothed centroid.
     */
    private data class TrackedState(
        val trackingId: Int,
        var smoothedCenterX: Float,
        var smoothedCenterY: Float,
        var lastBox: Rect,
        var label: String,
        var isSelected: Boolean = false,
        var missingFrames: Int = 0
    )

    init {
        val baseOptions = BaseOptions.builder()
            .setModelAssetPath(MODEL_PATH)
            .build()

        val options = ObjectDetector.ObjectDetectorOptions.builder()
            .setBaseOptions(baseOptions)
            .setMaxResults(MAX_RESULTS)
            .setScoreThreshold(SCORE_THRESHOLD)
            .setCategoryAllowlist(listOf(PERSON_CATEGORY))
            .build()

        objectDetector = ObjectDetector.createFromOptions(context, options)
        Log.d(TAG, "PersonDetectorProcessor initialized with EfficientDet-Lite0")
    }

    /**
     * Detect people in the given bitmap frame.
     * Returns a list of TrackedPerson with stable IDs across frames.
     */
    fun detectPersons(bitmap: Bitmap): List<TrackedPerson> {
        val mpImage = BitmapImageBuilder(bitmap).build()
        val result: ObjectDetectorResult = objectDetector.detect(mpImage)

        val currentDetections = result.detections()
            .filter { detection ->
                detection.categories().any {
                    it.categoryName().equals(PERSON_CATEGORY, ignoreCase = true)
                }
            }
            .map { detection ->
                val bbox = detection.boundingBox()
                val confidence = detection.categories()
                    .filter { it.categoryName().equals(PERSON_CATEGORY, ignoreCase = true) }
                    .maxOfOrNull { it.score() } ?: 0f

                DetectionBox(
                    rect = Rect(
                        bbox.left.toInt(),
                        bbox.top.toInt(),
                        bbox.right.toInt(),
                        bbox.bottom.toInt()
                    ),
                    confidence = confidence
                )
            }

        val imageWidth = bitmap.width.toFloat()
        val imageHeight = bitmap.height.toFloat()

        val matchedPersons = matchWithTrackedStates(currentDetections, imageWidth, imageHeight)
        return matchedPersons
    }

    /**
     * Match current detections with tracked states using multiple signals.
     */
    private fun matchWithTrackedStates(
        detections: List<DetectionBox>,
        imageWidth: Float,
        imageHeight: Float
    ): List<TrackedPerson> {

        if (trackedStates.isEmpty()) {
            // First frame — initialize all tracked states
            return detections.map { detection ->
                val id = nextTrackingId++
                val label = "Person " + (id + 1).toString()
                val cx = detection.rect.exactCenterX()
                val cy = detection.rect.exactCenterY()

                trackedStates[id] = TrackedState(
                    trackingId = id,
                    smoothedCenterX = cx,
                    smoothedCenterY = cy,
                    lastBox = detection.rect,
                    label = label
                )

                TrackedPerson(
                    trackingId = id,
                    boundingBox = detection.rect,
                    confidence = detection.confidence,
                    label = label
                )
            }
        }

        // Score all possible matches
        data class MatchCandidate(
            val stateId: Int,
            val detection: DetectionBox,
            val score: Float
        )

        val candidates = mutableListOf<MatchCandidate>()
        val maxHDist = imageWidth * MAX_HORIZONTAL_DIST_RATIO
        val maxVDist = imageHeight * MAX_VERTICAL_DIST_RATIO

        for ((stateId, state) in trackedStates) {
            for (detection in detections) {
                val detCx = detection.rect.exactCenterX()
                val detCy = detection.rect.exactCenterY()

                // 1. Horizontal distance score (most important — stays stable during exercises)
                val hDist = abs(state.smoothedCenterX - detCx)
                val hScore = (1f - (hDist / maxHDist).coerceIn(0f, 1f))

                // 2. Vertical distance score (more lenient)
                val vDist = abs(state.smoothedCenterY - detCy)
                val vScore = (1f - (vDist / maxVDist).coerceIn(0f, 1f))

                // 3. IoU score
                val iou = computeIoU(state.lastBox, detection.rect)

                // 4. Size similarity score (area ratio — prevents swapping between different-sized people)
                val prevArea = (state.lastBox.width() * state.lastBox.height()).toFloat()
                val detArea = (detection.rect.width() * detection.rect.height()).toFloat()
                val sizeRatio = if (prevArea > 0 && detArea > 0) {
                    minOf(prevArea, detArea) / maxOf(prevArea, detArea)
                } else 0f

                // Combined weighted score
                val combinedScore = HORIZONTAL_WEIGHT * hScore +
                        VERTICAL_WEIGHT * vScore +
                        IOU_WEIGHT * iou +
                        SIZE_WEIGHT * sizeRatio

                if (combinedScore >= MIN_MATCH_SCORE) {
                    candidates.add(MatchCandidate(stateId, detection, combinedScore))
                }
            }
        }

        // Greedy best-first matching
        candidates.sortByDescending { it.score }
        val matchedStateIds = mutableSetOf<Int>()
        val matchedDetections = mutableSetOf<DetectionBox>()
        val result = mutableListOf<TrackedPerson>()

        for (candidate in candidates) {
            if (candidate.stateId in matchedStateIds) continue
            if (candidate.detection in matchedDetections) continue

            val state = trackedStates[candidate.stateId] ?: continue
            val detCx = candidate.detection.rect.exactCenterX()
            val detCy = candidate.detection.rect.exactCenterY()

            // Update EMA-smoothed centroid
            state.smoothedCenterX = CENTROID_EMA_ALPHA * detCx + (1 - CENTROID_EMA_ALPHA) * state.smoothedCenterX
            state.smoothedCenterY = CENTROID_EMA_ALPHA * detCy + (1 - CENTROID_EMA_ALPHA) * state.smoothedCenterY
            state.lastBox = candidate.detection.rect
            state.missingFrames = 0

            result.add(
                TrackedPerson(
                    trackingId = state.trackingId,
                    boundingBox = candidate.detection.rect,
                    isSelected = state.isSelected,
                    confidence = candidate.detection.confidence,
                    label = state.label
                )
            )

            matchedStateIds.add(candidate.stateId)
            matchedDetections.add(candidate.detection)
        }

        // New detections — create new tracked states
        for (detection in detections) {
            if (detection in matchedDetections) continue

            val id = nextTrackingId++
            val label = "Person " + (id + 1).toString()
            val cx = detection.rect.exactCenterX()
            val cy = detection.rect.exactCenterY()

            trackedStates[id] = TrackedState(
                trackingId = id,
                smoothedCenterX = cx,
                smoothedCenterY = cy,
                lastBox = detection.rect,
                label = label
            )

            result.add(
                TrackedPerson(
                    trackingId = id,
                    boundingBox = detection.rect,
                    confidence = detection.confidence,
                    label = label
                )
            )
        }

        // Increment missing frame count for unmatched states, remove stale ones
        val staleIds = mutableListOf<Int>()
        for ((stateId, state) in trackedStates) {
            if (stateId !in matchedStateIds && stateId !in result.map { it.trackingId }) {
                state.missingFrames++
                // Selected person gets much longer grace period
                val maxMissing = if (state.trackingId == selectedTrackingId) {
                    SELECTED_MAX_MISSING_FRAMES
                } else {
                    MAX_MISSING_FRAMES
                }
                if (state.missingFrames > maxMissing) {
                    staleIds.add(stateId)
                }
            }
        }
        staleIds.forEach { trackedStates.remove(it) }

        return result
    }

    /**
     * Compute Intersection over Union between two rectangles.
     */
    private fun computeIoU(a: Rect, b: Rect): Float {
        val intersectLeft = maxOf(a.left, b.left)
        val intersectTop = maxOf(a.top, b.top)
        val intersectRight = minOf(a.right, b.right)
        val intersectBottom = minOf(a.bottom, b.bottom)

        if (intersectLeft >= intersectRight || intersectTop >= intersectBottom) {
            return 0f
        }

        val intersectionArea =
            (intersectRight - intersectLeft).toLong() * (intersectBottom - intersectTop).toLong()
        val areaA = (a.width().toLong()) * (a.height().toLong())
        val areaB = (b.width().toLong()) * (b.height().toLong())
        val unionArea = areaA + areaB - intersectionArea

        return if (unionArea > 0) intersectionArea.toFloat() / unionArea.toFloat() else 0f
    }

    fun close() {
        objectDetector.close()
        trackedStates.clear()
        Log.d(TAG, "PersonDetectorProcessor closed")
    }

    private data class DetectionBox(
        val rect: Rect,
        val confidence: Float
    )
}
