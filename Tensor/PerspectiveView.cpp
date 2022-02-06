//
//  PerspectiveView.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/1/29.
//

#include "PerspectiveView.hpp"

namespace otter {

void PerspectiveView::resizeSlowPath(const size_t newSize, const size_t oldSize) {
    if (newSize <= MAX_RANK_OF_TENSOR) {
        int64_t* tempStorage = outOflineStorage_;
        memcpy(&inlineStorage_[0], &tempStorage[0], MAX_RANK_OF_TENSOR * sizeof(inlineStorage_[0]));
        memcpy(&inlineStorage_[MAX_RANK_OF_TENSOR], &tempStorage[oldSize], MAX_RANK_OF_TENSOR * sizeof(inlineStorage_[0]));
        // CANNOT USE freeOutOfLineStorage() HERE! outOfLineStorage_
        // HAS BEEN OVERWRITTEN!
        // NOLINTNEXTLINE(cppcoreguidelines-no-malloc)
        free(tempStorage);
    } else {
        if (isInline()) {
            // CANNOT USE allocateOutOfLineStorage(newSize) HERE! WOULD
            // OVERWRITE inlineStorage_!
            // NOLINTNEXTLINE(cppcoreguidelines-no-malloc)
            int64_t* tempStorage = static_cast<int64_t*>(malloc(storageBytes(newSize)));
            const auto bytesToCopy = oldSize * sizeof(inlineStorage_[0]);
            const auto bytesToZero = (newSize > oldSize) ? (newSize - oldSize) * sizeof(tempStorage[0]) : 0;
            memcpy(&tempStorage[0], &inlineStorage_[0], bytesToCopy);
            if (bytesToZero) {
                memset(&tempStorage[oldSize], 0, bytesToZero);
            }
            memcpy(
                   &tempStorage[newSize],
                   &inlineStorage_[MAX_RANK_OF_TENSOR],
                   bytesToCopy);
            if (bytesToZero) {
                memset(&tempStorage[newSize + oldSize], 0, bytesToZero);
            }
            outOflineStorage_ = tempStorage;
        } else {
            const bool isGrowing = oldSize < newSize;
            if (isGrowing) {
                // Resize before shifting so that we have room.
                resizeOutOfLineStorage(newSize);
            }
            // Shift the old strides to their new starting point. Note
            // that this does not occur in the inline path above because
            // the stride starting point is not moving.
            memmove(outOflineStorage_ + newSize, outOflineStorage_ + oldSize, std::min(oldSize, newSize) * sizeof(outOflineStorage_[0]));
            if (!isGrowing) {
                // Resize after shifting so that we don't lose data.
                resizeOutOfLineStorage(newSize);
            } else {
                // Zero the end of the sizes portion.
                const auto bytesToZero = (newSize - oldSize) * sizeof(outOflineStorage_[0]);
                memset(&outOflineStorage_[oldSize], 0, bytesToZero);
                memset(&outOflineStorage_[newSize + oldSize], 0, bytesToZero);
            }
        }
    }
    size_ = newSize;
}

}   // end namespace otter
