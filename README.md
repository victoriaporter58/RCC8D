# RCC8D
A Python-based implementation [2] of the RCC8D ImageJ/Fiji plugin created by Gabriel Landini and David A Randell [1]. Read more about the ImageJ/Fiji plugin [here](https://blog.bham.ac.uk/intellimic/spatial-reasoning-with-imagej-using-the-region-connection-calculus/).

The purpose of this work is to provide a Python-based version of the ImageJ/Fiji plugin [1] that can be easily integrated with existing Python-based pipelines. It removes the need to install the ImageJ/Fiji software and does not require the user to understand how to use ImageJ/Fiji in order to run the RCC8D analysis. The RCC8D Python module is available [here](https://pypi.org/project/RCC8D/).

## Example Usage

This example computes the RCC8D relation of each ground truth mask with respect to all predicted masks made for the ground truth image. This is applicable for models such as the Segment Anything model which does not classify the masks that it generates.

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

## Citations

[1] Landini, G., Galton, A., and Randell, D. A. (2013). Discrete mereotopology for spatial reasoning in automated histological image analysis. IEEE Transactions on Pattern Analysis amp; Machine Intelligence, 35(03):568–581.

[2] Porter, V., Gault, R., Styles, I., and Curtis, T. (2024). Contextual Evaluation of Segmentation Models using Spatial Reasoning. Irish Machine Vision and Image Processing Conference (IMVIP).
