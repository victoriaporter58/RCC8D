import cv2
import numpy as np


class RCC8D:
    def __init__(self):
        self.relation_name = ["DR", "PO", "EQ", "PP", "PPi", "DC", "EC", "TPP",
                                    "NTPP", "TPPi", "NTPPi", "NCNTPP",
                                    "NCNTPPi", "NC", "PO*", "NTPP+", "NTPPi+",
                                    "DC+", "EC+", "DR_0", "Unknown"]

        self.attribute_name = ["non-empty", "b-close", "b-open", "regular",
                               "null-interior", "border", "atomic"]

    def update_border_attribute(self, img):
        return np.any(img[0, :]) or np.any(img[-1, :]) or np.any(img[:, 0]) or np.any(img[:, -1])

    def test_object(self, img, attributes, kernel):        
        # check if image is null-interior (attributes[4] = null-interior)
        eroded = cv2.erode(img, kernel, iterations=1)
        attributes[4] = not np.any(eroded)

        # check if image is b-open (attributes[2] = b-open)
        dilated = cv2.dilate(eroded, kernel, iterations=1)
        xor_open = cv2.bitwise_xor(img, dilated)
        attributes[2] = not np.any(xor_open)

        # check if image is b-close (attributes[1] = b-close)
        dilated = cv2.dilate(img, kernel, iterations=1)
        eroded = cv2.erode(dilated, kernel, iterations=1)
        xor_close = cv2.bitwise_xor(img, eroded)
        attributes[1] = not np.any(xor_close)

        # check if image is regular (attributes[3] = regular)
        xor_regular = cv2.bitwise_xor(dilated, eroded)
        attributes[3] = not np.any(xor_regular)       
        return attributes

    def sum_images(self, gt_im, pred_im):
        combined = cv2.add(gt_im, pred_im)  # pixel-wise addition of gt image and pred image
        hist = cv2.calcHist([combined], [0], None, [4], [0, 4])  # calculate histogram of the above addition, 4 bins corresponding to the pixel values 0, 1, 2, 3       
        return hist.flatten().astype(int)

    def compare_images(self, gt_img, pred_img, mode="RCC8D", attributes=False, details=False):
        # check images exist
        if gt_img is None or pred_img is None:
            return None

        # variable set-up
        relation_number = 19    # default relation (Null)
        attributes_gt = [False] * 7  # allocate space for the 7 attributes associated with the ground truth image
        attributes_pred = [False] * 7
        kernel = np.ones((3, 3), np.uint8)  # define kernel for morphological operation

        # prepare images for dilation
        # in terms of pixel values at this stage, we set the gt mask to 1, pred mask to 2, and background to 0
        gt_img = (gt_img // 255)  # convert pixel values to 0s and 1s
        pred_img = (pred_img // 255) * 2  # convert pixel values to 0s and 2s

        # add gt and pred together and compute a histogram using the result of the addition
        hist = self.sum_images(gt_img, pred_img)

        # in terms of pixel values at this stage, the background is 0, the gt mask is 1, pred mask is 2, and overlap of gt and pred masks is 3
        gt_bin_count, pred_bin_count, mask_overlap_bin_count = hist[1], hist[2], hist[3]

        # check if mask touches the border of the image (attributes_gt[5] = border)
        attributes_gt[5] = self.update_border_attribute(gt_img)
        attributes_pred[5] = self.update_border_attribute(pred_img)

        # check if gt mask is non-null (attributes_gt[0] = non-empty)
        if (gt_bin_count + mask_overlap_bin_count) != 0:
            attributes_gt[0] = True
            if attributes:
                self.test_object(gt_img, attributes_gt, kernel)

        # check if pred mask is non-null
        if (pred_bin_count + mask_overlap_bin_count) != 0:
            attributes_pred[0] = True
            if attributes:
                self.test_object(pred_img, attributes_pred, kernel)

        # check if gt mask is atomic (attributes_gt[6] = atomic)
        if (gt_bin_count == 1 and mask_overlap_bin_count == 0) or (gt_bin_count == 0 and mask_overlap_bin_count == 1):
            attributes_gt[6] = True

        # check if pred mask is atomic
        if (pred_bin_count == 1 and mask_overlap_bin_count == 0) or (pred_bin_count == 0 and mask_overlap_bin_count == 1):
            attributes_pred[6] = True

        # set default relation number
        if attributes_gt[0] and attributes_pred[0]:
            relation_number = 20  # default (unknown relation)

        # compute 5D relation number
        if gt_bin_count == 0 and pred_bin_count == 0 and mask_overlap_bin_count != 0:
            relation_number = 2  # EQ
        elif gt_bin_count != 0 and pred_bin_count != 0 and mask_overlap_bin_count != 0:
            relation_number = 1  # PO
        elif gt_bin_count == 0 and pred_bin_count != 0 and mask_overlap_bin_count != 0:
            relation_number = 3  # PP - used to determine TPP or NTPP
        elif gt_bin_count != 0 and pred_bin_count == 0 and mask_overlap_bin_count != 0:
            relation_number = 4  # PPi - used to determine TPPi or NTPPi
        elif gt_bin_count != 0 and pred_bin_count != 0 and mask_overlap_bin_count == 0:
            relation_number = 0  # DR - used to determine EC or DC

        # if necessary, compute more specific 8D relation number
        if relation_number == 3 and mode == "RCC8D":
            dilated_x = cv2.dilate(gt_img, kernel, iterations=1)
            hist = self.sum_images(dilated_x, pred_img)
            gt_bin_count, pred_bin_count, mask_overlap_bin_count = hist[1], hist[2], hist[3]           
            if gt_bin_count != 0:
                relation_number = 7  # TPP
            elif gt_bin_count == 0:
                relation_number = 8  # NTPP

        elif relation_number == 4 and mode == "RCC8D":
            dilated_y = cv2.dilate(pred_img, kernel, iterations=1)
            hist = self.sum_images(dilated_y, gt_img)
            gt_bin_count, pred_bin_count, mask_overlap_bin_count = hist[1], hist[2], hist[3]           
            if pred_bin_count != 0:
                relation_number = 9  # TPPi
            elif pred_bin_count == 0:
                relation_number = 10  # NTPPi

        elif relation_number == 0 and mode == "RCC8D":
            dilated_x = cv2.dilate(gt_img, kernel, iterations=1)
            hist = self.sum_images(dilated_x, pred_img)
            gt_bin_count, pred_bin_count, mask_overlap_bin_count = hist[1], hist[2], hist[3]            
            if mask_overlap_bin_count != 0:
                relation_number = 6  # EC
            elif mask_overlap_bin_count == 0:
                relation_number = 5  # DC

        results = {
            "relation_name": self.relation_name[relation_number],
            "relation_number": relation_number,
        }

        if attributes:
            results.update({
                "attributes_x": {name: value for name, value in zip(self.attribute_name, attributes_gt)},
                "attributes_y": {name: value for name, value in zip(self.attribute_name, attributes_pred)},
            })

        if details:
            results.update({
                "image_x_sum": gt_img,
                "image_y_sum": pred_img,
                "histogram": hist,
            })

        return results
