from rcc8d import RCC8D
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
    rcc8d = RCC8D()

    for gt in os.listdir(path_to_gt_images):
        gt_img_path = os.path.join(path_to_gt_images, gt)
        gt_img = cv2.imread(gt_img_path, cv2.IMREAD_GRAYSCALE)
        eval_out_json[gt_img_path] = {}

        print("Computing RCC8D relations for " + gt)

        for pred in os.listdir(path_to_pred_images):
            if ".png" in pred:
                pred_img_path = os.path.join(path_to_pred_images, pred)
                pred_img = cv2.imread(pred_img_path, cv2.IMREAD_GRAYSCALE)
                
                result = rcc8d.compare_images(gt_img, pred_img, mode="RCC8D", attributes=True, details=True)
                
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