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
 * 2. Draws the blurred copy over the entire canvas using the overlay's transformation matrix
 * 3. Clips out the selected person's region using a segmentation mask so the SHARP camera preview shows through
 */
class BlurOverlayGraphic(
    overlay: GraphicOverlay,
    private val selectedBox: Rect,
    private val cameraBitmap: Bitmap,
    private val segmentationMaskBitmap: Bitmap? = null,
    private val cropRect: Rect? = null
) : GraphicOverlay.Graphic(overlay) {

    private val bitmapPaint = Paint().apply {
        isAntiAlias = true
        isFilterBitmap = true
        alpha = 160 // Reduced opacity (~60%) so the sharp camera preview subtly shows through
    }

    private val erasePaint = Paint().apply {
        isAntiAlias = true
        isFilterBitmap = true
        xfermode = android.graphics.PorterDuffXfermode(android.graphics.PorterDuff.Mode.DST_OUT)
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

        // We use a hardware layer so PorterDuff modes blend against this layer, not the whole screen
        canvas.saveLayer(null, null)

        // Draw blurred frame over the canvas
        canvas.drawBitmap(blurred, transformationMatrix, bitmapPaint)

        if (segmentationMaskBitmap != null && cropRect != null && !segmentationMaskBitmap.isRecycled) {
            // Calculate where this mask should be drawn on the canvas
            // The cropRect is in the original cameraBitmap coordinates.
            // We need to map cropRect to canvas coordinates using transformationMatrix.
            val maskMatrix = android.graphics.Matrix()
            
            // First, scale and translate the mask so it fits its position in the original image
            maskMatrix.postTranslate(cropRect.left.toFloat(), cropRect.top.toFloat())
            
            // Then apply the global overlay transformation
            maskMatrix.postConcat(transformationMatrix)

            // Erase the blurred pixels where the mask is drawn
            canvas.drawBitmap(segmentationMaskBitmap, maskMatrix, erasePaint)
        } else {
            // Fallback: Body-shaped cutout with padding and rounded corners
            val left = translateX(selectedBox.left.toFloat())
            val top = translateY(selectedBox.top.toFloat())
            val right = translateX(selectedBox.right.toFloat())
            val bottom = translateY(selectedBox.bottom.toFloat())

            val drawLeft = minOf(left, right)
            val drawRight = maxOf(left, right)
            val drawTop = minOf(top, bottom)
            val drawBottom = maxOf(top, bottom)

            val cutoutRect = RectF(
                drawLeft - CUTOUT_PADDING,
                drawTop - CUTOUT_PADDING,
                drawRight + CUTOUT_PADDING,
                drawBottom + CUTOUT_PADDING
            )

            val rx = cutoutRect.width() * 0.22f
            val ry = cutoutRect.height() * 0.08f
            
            // Draw a rounded rectangle with erasePaint to cut a hole into the blurred background
            canvas.drawRoundRect(cutoutRect, rx, ry, erasePaint)
        }

        canvas.restore()

        // Clean up
        if (!blurred.isRecycled) blurred.recycle()
    }

    companion object {
        private const val CUTOUT_PADDING = 16f
    }
}
