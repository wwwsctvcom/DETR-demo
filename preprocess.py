import os
import VIA2COCO.convert as via2coco
from loguru import logger
import numpy as np
from PIL import Image, ImageDraw
from utils.dataset import CocoDetection, CollateFn
from transformers import DetrConfig, DetrForObjectDetection, DetrImageProcessor

if __name__ == "__main__":
    # convert the data to coco format

    data_path = './data/balloon/'

    first_class_index = 0

    for keyword in ['train', 'val']:
        input_dir = data_path + keyword + '/'
        input_json = input_dir + 'via_region_data.json'
        categories = ['balloon']
        super_categories = ['N/A']
        output_json = input_dir + 'custom_' + keyword + '.json'

        print('Converting {} from VIA format to COCO format'.format(input_json))

        coco_dict = via2coco.convert(
            imgdir=input_dir,
            annpath=input_json,
            categories=categories,
            super_categories=super_categories,
            output_file_name=output_json,
            first_class_index=first_class_index,
        )

    # visualize the data
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

    train_dataset = CocoDetection(img_folder='./data/balloon/train', processor=processor)

    # based on https://github.com/woctezuma/finetune-detr/blob/master/finetune_detr.ipynb
    image_ids = train_dataset.coco.getImgIds()
    # let's pick a random image
    image_id = image_ids[np.random.randint(0, len(image_ids))]
    logger.info('Image n°{}'.format(image_id))
    image = train_dataset.coco.loadImgs(image_id)[0]
    image = Image.open(os.path.join('./data/balloon/train', image['file_name']))

    annotations = train_dataset.coco.imgToAnns[image_id]
    draw = ImageDraw.Draw(image, "RGBA")

    cats = train_dataset.coco.cats
    id2label = {k: v['name'] for k, v in cats.items()}

    for annotation in annotations:
        box = annotation['bbox']
        class_idx = annotation['category_id']
        x, y, w, h = tuple(box)
        draw.rectangle((x, y, x + w, y + h), outline='red', width=1)
        draw.text((x, y), id2label[class_idx], fill='white')
    image.show()
