import io
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from keras.models import load_model, model_from_json
from scipy.ndimage import gaussian_filter, maximum_filter
from matplotlib.backends.backend_agg import RendererAgg

from lib.constants import *


class AppHelper():
    def __init__(self, model_weights, model_json) -> None:
        self._load_model(model_json=model_json, model_weights=model_weights)

    def predict_in_memory(self, img, visualize_scatter=True, visualize_skeleton=True, average_flip_prediction=True):
        X_batch, y_stacked = self.load_and_preprocess_img(img, 1)
        y_batch = y_stacked[0]  # take first hourglass section
        img_id_batch = None

        return self._predict_and_visualize(
            X_batch,
            visualize_scatter=visualize_scatter,
            visualize_skeleton=visualize_skeleton,
            average_flip_prediction=average_flip_prediction
        )

    def load_and_preprocess_img(self, img_path, num_hg_blocks, bbox=None):
        img = Image.open(img_path).convert('RGB')

        if bbox is not None:
            # If a bounding box is provided, use it
            bbox = np.array(bbox, dtype=int)

            # Crop with box of order left, upper, right, lower
            img = img.crop(box=bbox)

        img = img.resize((256, 256), Image.BICUBIC)

        new_img = np.array(img)

        # Add a 'batch' axis
        X_batch = np.expand_dims(new_img.astype('float'), axis=0)

        # Add dummy heatmap "ground truth," duplicated 'num_hg_blocks' times
        y_batch = [np.zeros((1, *(OUTPUT_DIM), NUM_COCO_KEYPOINTS), dtype='float') for _ in range(num_hg_blocks)]

        # Normalize input image
        X_batch /= 255
        return X_batch, y_batch

    def _load_model(self, model_json, model_weights):
        with open(model_json) as f:
            self.model = model_from_json(f.read())
        self.model.load_weights(model_weights)

    def _predict_and_visualize(self, X_batch, visualize_scatter=True, visualize_skeleton=True, average_flip_prediction=True):
        predicted_heatmaps_batch = self.predict_heatmaps(X_batch)

        if visualize_scatter or visualize_skeleton:
            keypoints_batch = self._batch(predicted_heatmaps_batch)

            if average_flip_prediction:
                # Average predictions from original image and the untransformed flipped image to get a more accurate prediction
                predicted_heatmaps_batch_2 = self.predict_heatmaps(X_batch=X_batch, predict_using_flip=True)

                keypoints_batch_2 = self.heatmaps_to_keypoints_batch(predicted_heatmaps_batch_2)

                for i in range(keypoints_batch.shape[0]):
                    # Average predictions from normal and flipped input
                    keypoints_batch[i] = self._average_LR_flip_predictions(
                        keypoints_batch[i], keypoints_batch_2[i], coco_format=False
                    )
            return self.visualize_keypoints(X_batch, keypoints_batch, show_skeleton=visualize_skeleton)

    def _average_LR_flip_predictions(self, prediction_1, prediction_2, coco_format=True):
        # Average predictions from original image and the untransformed flipped image to get a more accurate prediction
        original_shape = prediction_1.shape

        prediction_1_flat = prediction_1.flatten()
        prediction_2_flat = prediction_2.flatten()

        output_prediction = prediction_1_flat

        for j in range(NUM_COCO_KEYPOINTS):
            base = j * NUM_COCO_KP_ATTRBS

            n = 0
            x_sum = 0
            y_sum = 0
            vc_sum = 0  # Could be visibility or confidence

            # Verify visibility flag
            if prediction_1_flat[base + 2] >= HM_TO_KP_THRESHOLD:
                x_sum += prediction_1_flat[base]
                y_sum += prediction_1_flat[base + 1]
                vc_sum += prediction_1_flat[base + 2]
                n += 1

            if prediction_2_flat[base + 2] >= HM_TO_KP_THRESHOLD:
                x_sum += prediction_2_flat[base]
                y_sum += prediction_2_flat[base + 1]
                vc_sum += prediction_2_flat[base + 2]
                n += 1

            # Verify that no division by 0 will occur
            if n > 0:
                output_prediction[base] = round(x_sum / n)
                output_prediction[base + 1] = round(y_sum / n)
                output_prediction[base + 2] = 1 if coco_format else round(vc_sum / n)

        if not coco_format:
            output_prediction = np.reshape(output_prediction, original_shape)

        return output_prediction

    def predict_heatmaps(self, X_batch, predict_using_flip=False):
        def _predict(X_batch):
            # Instead of calling model.predict or model.predict_on_batch, we call model by itself.
            # See https://stackoverflow.com/questions/66271988/warningtensorflow11-out-of-the-last-11-calls-to-triggered-tf-function-retracin
            # This should fix our memory leak in keras
            return np.array(self.model.predict_on_batch(X_batch))

        # X_batch has dimensions (batch, x, y, channels)
        # Run both original and flipped image through and average the predictions
        # Typically increases accuracy by a few percent
        if predict_using_flip:
            # Horizontal flip each image in batch
            X_batch_flipped = X_batch[:, :, ::-1, :]

            # Feed flipped image into model
            # output shape is (num_hg_blocks, X_batch_size, 64, 64, 17)
            predicted_heatmaps_batch_flipped = _predict(X_batch_flipped)

            # indices to flip order of Left and Right heatmaps [0, 2, 1, 4, 3, 6, 5, 8, 7, etc]
            reverse_LR_indices = [0] + [2 * x - y for x in range(1, 9) for y in range(2)]

            # reverse horizontal flip AND reverse left/right heatmaps
            predicted_heatmaps_batch = predicted_heatmaps_batch_flipped[:, :, :, ::-1, reverse_LR_indices]
        else:
            predicted_heatmaps_batch = _predict(X_batch)

        return predicted_heatmaps_batch

    def visualize_keypoints(self, X_batch, keypoints_batch, show_skeleton=True):
        for i in range(len(X_batch)):
            X = X_batch[i]
            keypoints = keypoints_batch[i]

            fig, ax = plt.subplots(figsize=(X.shape[1] / 100, X.shape[0] / 100), dpi=100)

            # Plot predicted keypoints on bounding box image
            x_left = []
            y_left = []
            x_right = []
            y_right = []
            valid = np.zeros(NUM_COCO_KEYPOINTS)

            for i in range(NUM_COCO_KEYPOINTS):
                if keypoints[i, 0] != 0 and keypoints[i, 1] != 0:
                    valid[i] = 1

                    if i % 2 == 0:
                        x_right.append(keypoints[i, 0])
                        y_right.append(keypoints[i, 1])
                    else:
                        x_left.append(keypoints[i, 0])
                        y_left.append(keypoints[i, 1])

            if show_skeleton:
                for i in range(len(COCO_SKELETON)):
                    # joint a to joint b
                    a = COCO_SKELETON[i, 0]
                    b = COCO_SKELETON[i, 1]

                    # if both are valid keypoints
                    if valid[a] and valid[b]:
                        ax.plot(
                            [keypoints[a, 0], keypoints[b, 0]],
                            [keypoints[a, 1], keypoints[b, 1]],
                            color=COLOUR_MAP[i]
                        )

            ax.scatter(x_left, y_left, color=COLOUR_MAP[0])
            ax.scatter(x_right, y_right, color=COLOUR_MAP[4])
            ax.axis('off')
            ax.imshow(X)

            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(
                buf,
                format='png',
                bbox_inches='tight',
                pad_inches=0,
                transparent=False,
                dpi=100
            )
            plt.close(fig)

            buf.seek(0)
            im = Image.open(buf).convert('RGB')
            visualized = np.array(im)
            buf.close()
            im.close()

            # Remove padding
            visualized = self.remove_padding(visualized)
            return visualized

    def heatmaps_to_keypoints_batch(self, heatmaps_batch, threshold=HM_TO_KP_THRESHOLD):
        keypoints_batch = []

        # dimensions are (num_hg_blocks, batch, x, y, keypoint)
        for i in range(heatmaps_batch.shape[1]):
            # Get predicted keypoints from last hourglass (last element)
            keypoints = self.heatmaps_to_keypoints(heatmaps_batch[-1, i, :, :, :])

            keypoints_batch.append(keypoints)

        return np.array(keypoints_batch)

    def heatmaps_to_keypoints(self, heatmaps, threshold=HM_TO_KP_THRESHOLD):
        keypoints = np.zeros((NUM_COCO_KEYPOINTS, NUM_COCO_KP_ATTRBS))
        for i in range(NUM_COCO_KEYPOINTS):
            hmap = heatmaps[:, :, i]
            # Resize heatmap from Output DIM to Input DIM
            resized_hmap = cv2.resize(hmap, INPUT_DIM, interpolation=cv2.INTER_LINEAR)
            # Do a heatmap blur with gaussian_filter
            resized_hmap = gaussian_filter(resized_hmap, REVERSE_HEATMAP_SIGMA)
            peaks = self._non_max_suppression(resized_hmap, threshold, windowSize=3)
            y, x = np.unravel_index(np.argmax(peaks), peaks.shape)
            if peaks[y, x] > HM_TO_KP_THRESHOLD_POST_FILTER:
                conf = peaks[y, x]
            else:
                x, y, conf = 0, 0, 0

            keypoints[i, 0] = x
            keypoints[i, 1] = y
            keypoints[i, 2] = conf

        return keypoints

    def _non_max_suppression(self, plain, threshold, windowSize=3):
        # Clear values less than threshold
        under_thresh_indices = plain < threshold
        plain[under_thresh_indices] = 0
        return plain * (plain == maximum_filter(plain, footprint=np.ones((windowSize, windowSize))))

    def remove_padding(self, image):
        # Find non-zero indices in the image
        non_zero_indices = np.nonzero(image)

        # Get the bounding box coordinates of the non-zero region
        min_x = np.min(non_zero_indices[1])
        max_x = np.max(non_zero_indices[1])
        min_y = np.min(non_zero_indices[0])
        max_y = np.max(non_zero_indices[0])

        # Crop the image to the bounding box
        cropped_image = image[min_y:max_y, min_x:max_x, :]

        return cropped_image

