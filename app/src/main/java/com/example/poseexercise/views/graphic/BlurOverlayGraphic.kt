package com.example.poseexercise.views.graphic

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.Path
import android.graphics.Rect
import android.graphics.RectF
import com.example.poseexercise.util.FastBlurHelper

/**
 * Portrait-mode blur overlay:
 *
 * 1. Takes the camera frame bitmap and creates a blurred copy
 * 2. Draws the blurred copy over the entire canvas using the overlay's
 *    transformation matrix (handles scaling, offset, mirroring)
 * 3. Clips out the selected person's region so the SHARP camera preview
 *    underneath shows through
 *
 * Result: selected person is crisp, everything else has depth-of-field blur.
 */
class BlurOverlayGraphic(
    overlay: GraphicOverlay,
    private val selectedBox: Rect,
    private val cameraBitmap: Bitmap
) : GraphicOverlay.Graphic(overlay) {

    private val bitmapPaint = Paint().apply {
        isAntiAlias = true
        isFilterBitmap = true
    }

    override fun draw(canvas: Canvas) {
        if (cameraBitmap.isRecycled) return

        // Create blurred version of the camera frame
        val blurred = try {
            FastBlurHelper.blur(cameraBitmap)
        } catch (e: Exception) {
            return
        }
        if (blurred.isRecycled) return

        // Map selected person's bounding box from image coords to canvas coords
        val left = translateX(selectedBox.left.toFloat())
        val top = translateY(selectedBox.top.toFloat())
        val right = translateX(selectedBox.right.toFloat())
        val bottom = translateY(selectedBox.bottom.toFloat())

        // Correct for front-camera flip (translateX may reverse left/right)
        val drawLeft = minOf(left, right)
        val drawRight = maxOf(left, right)
        val drawTop = minOf(top, bottom)
        val drawBottom = maxOf(top, bottom)

        // Body-shaped cutout with padding and rounded corners
        val cutoutRect = RectF(
            drawLeft - CUTOUT_PADDING,
            drawTop - CUTOUT_PADDING,
            drawRight + CUTOUT_PADDING,
            drawBottom + CUTOUT_PADDING
        )

        // Build clip path: full canvas minus the person cutout
        val fullPath = Path()
        fullPath.addRect(
            0f, 0f,
            canvas.width.toFloat(), canvas.height.toFloat(),
            Path.Direction.CW
        )

        val cutoutPath = Path()
        val rx = cutoutRect.width() * 0.22f
        val ry = cutoutRect.height() * 0.08f
        cutoutPath.addRoundRect(cutoutRect, rx, ry, Path.Direction.CW)

        fullPath.op(cutoutPath, Path.Op.DIFFERENCE)

        // Draw blurred frame everywhere except the cutout
        canvas.save()
        canvas.clipPath(fullPath)
        canvas.drawBitmap(blurred, transformationMatrix, bitmapPaint)
        canvas.restore()

        // Clean up
        if (!blurred.isRecycled) blurred.recycle()
    }

    companion object {
        private const val CUTOUT_PADDING = 16f
    }
}
