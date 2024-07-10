# RCC8D

A Python-based implementation [2] of the RCC8D ImageJ/Fiji plugin created by Gabriel Landini and David A Randell [1]. Read more about the ImageJ/Fiji plugin [here](https://blog.bham.ac.uk/intellimic/spatial-reasoning-with-imagej-using-the-region-connection-calculus/).

The purpose of this work is to provide a Python-based version of the ImageJ/Fiji plugin [1] that can be easily integrated with existing Python-based pipelines. It removes the need to install the ImageJ/Fiji software and does not require the user to understand how to use ImageJ/Fiji in order to run the RCC8D analysis. The RCC8D Python module is available [here](https://pypi.org/project/RCC8D/).

## Input Structure

To run the RCC8D analysis, the ground truth images and predicted masks must be presented according to the file structure below. If a raw image has multiple ground truths, they must be encapsulated within the same folder. All predictions made for a raw image must be stored in the same folder which should have the same name as the raw image file e.g., raw_image_0.tif would have a ground truth folder and a predictions folder called raw_image_0.

```bash
test_images:
    ├───gt
        ├───raw_image_0
            ├───ground_truth_0.png
            ├───ground_truth_1.png
            ├───ground_truth_2.png
            ├───ground_truth_3.png
        ├───raw_image_1
            ├───ground_truth_0.png
            ├───ground_truth_1.png
            ├───ground_truth_2.png
    └───pred
        ├───raw_image_0
            ├───prediction_0.png
            ├───prediction_1.png
            ├───prediction_2.png
            ├───prediction_3.png
            ├───prediction_4.png
            ├───prediction_5.png
        ├───raw_image_1
            ├───prediction_0.png
            ├───prediction_1.png
```

## Example Usage

This example computes the RCC8D relation of each ground truth mask with respect to all predicted masks made for the ground truth image. This is applicable for models such as the Segment Anything model which does not classify the masks that it generates. This script can be modified to suit the requirements of other image segmentation models.

```python
from RCC8D.rcc8d import RCC8D
import os
import cv2
import json


def main():
    # argument set-up
    path_to_gt_images = "test_images/gt"
    path_to_pred_images = "test_images/pred"
    eval_out_path = os.path.join("evaluation", "RCC8D_output.json")
    eval_out_json = {}

    # initialise RCC8D class
    rcc8d_ = RCC8D()

    # compute RCC8D relations for every ground truth mask W.R.T every predicted mask
    for gt in os.listdir(path_to_gt_images):
        gt_img_path = os.path.join(path_to_gt_images, gt)
        gt_img = cv2.imread(gt_img_path, cv2.IMREAD_GRAYSCALE)
        eval_out_json[gt_img_path] = {}

        print("Computing RCC8D relations for " + gt)

        for pred in os.listdir(path_to_pred_images):
            if ".png" in pred:
                pred_img_path = os.path.join(path_to_pred_images, pred)
                pred_img = cv2.imread(pred_img_path, cv2.IMREAD_GRAYSCALE)

                result = rcc8d_.compare_images(gt_img, pred_img, mode="RCC8D", attributes=True, details=True)

                eval_out_json[gt_img_path][pred_img_path] = {
                    "gt_attributes": str(result["attributes_x"]),
                    "pred_attributes": str(result["attributes_y"]),
                    "relation_name": result["relation_name"],
                    "relation_number": str(result["relation_number"])
                }

    # write the results to a JSON file
    with open(eval_out_path, 'w') as json_file:
        json.dump(eval_out_json, json_file, indent=4)

    print("Done!")


if __name__ == "__main__":
    main()

```

To compute the relationship between a single ground truth mask and a single prediction, the following line would be used:

```python
result = rcc8d_.compare_images(gt_img, pred_img, mode="RCC8D", attributes=True, details=True)
```

## Example Output

The following JSON structure demonstrates the output for a raw image, gt_0.tif, which has 2 ground truth masks, ground_truth_0.png and ground_truth_1.png, with respect to 3 predicted masks, pred_0.png, pred_1.png, and pred_2.png. This output can be processed to count the proportion of relation types per image (or per ground truth class) to illustrate a model's tendency to over- or underestimate.

```json
{
    "test_images/gt/gt_0/ground_truth_0.png": {
        "test_images/pred\\pred_0.png": {
            "gt_attributes": "{'non-empty': True, 'b-close': True, 'b-open': True, 'regular': False, 'null-interior': False, 'border': False, 'atomic': False}",
            "pred_attributes": "{'non-empty': True, 'b-close': True, 'b-open': True, 'regular': False, 'null-interior': False, 'border': False, 'atomic': False}",
            "relation_name": "DC",
            "relation_number": "5"
        },
        "test_images/pred\\pred_1.png": {
            "gt_attributes": "{'non-empty': True, 'b-close': True, 'b-open': True, 'regular': False, 'null-interior': False, 'border': False, 'atomic': False}",
            "pred_attributes": "{'non-empty': True, 'b-close': True, 'b-open': True, 'regular': False, 'null-interior': False, 'border': True, 'atomic': False}",
            "relation_name": "EC",
            "relation_number": "6"
        },
        "test_images/pred\\pred_2.png": {
            "gt_attributes": "{'non-empty': True, 'b-close': True, 'b-open': True, 'regular': False, 'null-interior': False, 'border': False, 'atomic': False}",
            "pred_attributes": "{'non-empty': True, 'b-close': True, 'b-open': True, 'regular': False, 'null-interior': False, 'border': False, 'atomic': False}",
            "relation_name": "NTPP",
            "relation_number": "8"
        }
    },
    "test_images/gt/gt_0/ground_truth_1.png": {
        "test_images/pred\\pred_0.png": {
            "gt_attributes": "{'non-empty': True, 'b-close': True, 'b-open': True, 'regular': False, 'null-interior': False, 'border': False, 'atomic': False}",
            "pred_attributes": "{'non-empty': True, 'b-close': True, 'b-open': True, 'regular': False, 'null-interior': False, 'border': False, 'atomic': False}",
            "relation_name": "NTPPi",
            "relation_number": "10"
        },
        "test_images/pred\\pred_1.png": {
            "gt_attributes": "{'non-empty': True, 'b-close': True, 'b-open': True, 'regular': False, 'null-interior': False, 'border': False, 'atomic': False}",
            "pred_attributes": "{'non-empty': True, 'b-close': True, 'b-open': True, 'regular': False, 'null-interior': False, 'border': True, 'atomic': False}",
            "relation_name": "PO",
            "relation_number": "1"
        },
        "test_images/pred\\pred_2.png": {
            "gt_attributes": "{'non-empty': True, 'b-close': True, 'b-open': True, 'regular': False, 'null-interior': False, 'border': False, 'atomic': False}",
            "pred_attributes": "{'non-empty': True, 'b-close': False, 'b-open': True, 'regular': False, 'null-interior': False, 'border': True, 'atomic': False}",
            "relation_name": "TPP",
            "relation_number": "7"
        }
    }
}
```

## Citations

[1] Landini, G., Galton, A., and Randell, D. A. (2013). Discrete mereotopology for spatial reasoning in automated histological image analysis. IEEE Transactions on Pattern Analysis amp; Machine Intelligence, 35(03):568–581.

[2] Porter, V., Gault, R., Styles, I., and Curtis, T. (2024). Contextual Evaluation of Segmentation Models using Spatial Reasoning. Irish Machine Vision and Image Processing Conference (IMVIP).
