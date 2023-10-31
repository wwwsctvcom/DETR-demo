import torch
from utils.tools import get_cur_time
from utils.dataset import CocoDetection, CollateFn
from coco_eval import CocoEvaluator
from tqdm.notebook import tqdm
from torch.utils.data import DataLoader
from transformers import DetrImageProcessor, DetrForObjectDetection


def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)

def prepare_for_coco_detection(predictions):
    coco_results = []
    for original_id, prediction in predictions.items():
        if len(prediction) == 0:
            continue

        boxes = prediction["boxes"]
        boxes = convert_to_xywh(boxes).tolist()
        scores = prediction["scores"].tolist()
        labels = prediction["labels"].tolist()

        coco_results.extend(
            [
                {
                    "image_id": original_id,
                    "category_id": labels[k],
                    "bbox": box,
                    "score": scores[k],
                }
                for k, box in enumerate(boxes)
            ]
        )
    return coco_results



if __name__ == "__main__":
    # loading model and processor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DetrForObjectDetection.from_pretrained(get_cur_time() + "/detr-finetuned-balloon", id2label={0: "balloon"})
    model.to(device)
    processor = DetrImageProcessor.from_pretrained(get_cur_time() + "/detr-finetuned-balloon")

    # loading dataset
    val_dataset = CocoDetection(img_folder='./data/balloon/val', processor=processor, train=False)
    collate_fn = CollateFn(processor)
    val_dataloader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=2)

    # initialize evaluator with ground truth (gt)
    evaluator = CocoEvaluator(coco_gt=val_dataset.coco, iou_types=["bbox"])

    print("Running evaluation...")
    for idx, batch in enumerate(tqdm(val_dataloader)):
        # get the inputs
        pixel_values = batch["pixel_values"].to(device)
        pixel_mask = batch["pixel_mask"].to(device)
        labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]] # these are in DETR format, resized + normalized

        # forward pass
        with torch.no_grad():
          outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)

        # turn into a list of dictionaries (one item for each example in the batch)
        orig_target_sizes = torch.stack([target["orig_size"] for target in labels], dim=0)
        results = processor.post_process_object_detection(outputs, target_sizes=orig_target_sizes)

        # provide to metric
        # metric expects a list of dictionaries, each item
        # containing image_id, category_id, bbox and score keys
        predictions = {target['image_id'].item(): output for target, output in zip(labels, results)}
        predictions = prepare_for_coco_detection(predictions)
        evaluator.update(predictions)

    evaluator.synchronize_between_processes()
    evaluator.accumulate()
    evaluator.summarize()
