package com.example.poseexercise.views.graphic

import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect

/**
 * Draws a bounding box around a detected person with their tracking ID label.
 * Selected person gets a bright green border; non-selected get a gray border.
 */
class PersonBoundingBoxGraphic(
    overlay: GraphicOverlay,
    private val boundingBox: Rect,
    private val trackingId: Int,
    private val label: String,
    private val isSelected: Boolean,
    private val confidence: Float
) : GraphicOverlay.Graphic(overlay) {

    private val boxPaint = Paint().apply {
        style = Paint.Style.STROKE
        strokeWidth = if (isSelected) 6f else 3f
        color = if (isSelected) COLOR_SELECTED else COLOR_UNSELECTED
    }

    private val labelBackgroundPaint = Paint().apply {
        style = Paint.Style.FILL
        color = if (isSelected) COLOR_SELECTED_BG else COLOR_UNSELECTED_BG
    }

    private val labelTextPaint = Paint().apply {
        color = Color.WHITE
        textSize = LABEL_TEXT_SIZE
        isFakeBoldText = isSelected
    }

    override fun draw(canvas: Canvas) {
        // Transform coordinates from image space to view space
        val left = translateX(boundingBox.left.toFloat())
        val top = translateY(boundingBox.top.toFloat())
        val right = translateX(boundingBox.right.toFloat())
        val bottom = translateY(boundingBox.bottom.toFloat())

        // Draw bounding box
        canvas.drawRect(left, top, right, bottom, boxPaint)

        // Draw label background
        val labelText = if (isSelected) "$label ✓" else "$label"
        val labelWidth = labelTextPaint.measureText(labelText)
        val labelHeight = LABEL_TEXT_SIZE + LABEL_PADDING * 2

        canvas.drawRect(
            left,
            top - labelHeight,
            left + labelWidth + LABEL_PADDING * 2,
            top,
            labelBackgroundPaint
        )

        // Draw label text
        canvas.drawText(
            labelText,
            left + LABEL_PADDING,
            top - LABEL_PADDING,
            labelTextPaint
        )

        // Draw confidence underneath the box (small text)
        if (confidence > 0f) {
            val confText = String.format("%.0f%%", confidence * 100)
            val confPaint = Paint().apply {
                color = if (isSelected) COLOR_SELECTED else COLOR_UNSELECTED
                textSize = CONFIDENCE_TEXT_SIZE
            }
            canvas.drawText(confText, left + LABEL_PADDING, bottom + CONFIDENCE_TEXT_SIZE + 4f, confPaint)
        }
    }

    companion object {
        private const val LABEL_TEXT_SIZE = 36f
        private const val CONFIDENCE_TEXT_SIZE = 24f
        private const val LABEL_PADDING = 8f
        private val COLOR_SELECTED = Color.rgb(76, 175, 80)         // Material Green
        private val COLOR_UNSELECTED = Color.rgb(158, 158, 158)     // Material Gray
        private val COLOR_SELECTED_BG = Color.argb(200, 46, 125, 50)  // Dark green BG
        private val COLOR_UNSELECTED_BG = Color.argb(160, 66, 66, 66) // Dark gray BG
    }
}
