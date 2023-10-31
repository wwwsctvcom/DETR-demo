import os
import torch
from utils.dataset import CocoDetection, CollateFn
from PIL import Image
from loguru import logger
from utils.tools import get_cur_time
from torch.utils.data import DataLoader
from transformers import DetrImageProcessor, DetrForObjectDetection

import matplotlib.pyplot as plt

# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]


def plot_results(pil_img, scores, labels, boxes):
    plt.figure(figsize=(16, 10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    for score, label, (xmin, ymin, xmax, ymax), c in zip(scores.tolist(), labels.tolist(), boxes.tolist(), colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        text = f'{model.config.id2label[label]}: {score:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.show()


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

    pixel_values, target = val_dataset[1]

    pixel_values = pixel_values.unsqueeze(0).to(device)
    logger.info(pixel_values.shape)

    with torch.no_grad():
        # forward pass to get class logits and bounding boxes
        outputs = model(pixel_values=pixel_values, pixel_mask=None)
    logger.info("Outputs:", outputs.keys())

    # load image based on ID
    image_id = target['image_id'].item()
    image = val_dataset.coco.loadImgs(image_id)[0]
    image = Image.open(os.path.join('/kaggle/working/balloon/val', image['file_name']))

    # postprocess model outputs
    width, height = image.size
    postprocessed_outputs = processor.post_process_object_detection(outputs,
                                                                    target_sizes=[(height, width)],
                                                                    threshold=0.9)
    results = postprocessed_outputs[0]
    plot_results(image, results['scores'], results['labels'], results['boxes'])
