package com.example.poseexercise.data

import android.graphics.Rect

/**
 * Data class representing a detected and tracked person in the camera frame.
 *
 * @param trackingId Stable ID assigned to this person for cross-frame tracking
 * @param boundingBox Bounding box of the person in image coordinates
 * @param isSelected Whether this person is currently selected for pose tracking
 * @param label Display label (e.g., "Person 1")
 * @param confidence Detection confidence score (0.0 - 1.0)
 */
data class TrackedPerson(
    val trackingId: Int,
    val boundingBox: Rect,
    val isSelected: Boolean = false,
    val label: String = "Person $trackingId",
    val confidence: Float = 0f
)
