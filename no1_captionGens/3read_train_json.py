
import json

filename = '/home/lin7u/Downloads/dataset/coco2014/annotations/captions_val2014.json'

# captions_train2014.json
# dict_keys(['info', 'images', 'licenses', 'annotations'])

#info = {'description': 'COCO Dataset', 'url': '.org', 'version': '1.0',
#           'year':, 'contributor': 'COCO Consortium', 'date_created': '2017'}
# images = [{license': 5, 'file_name': '.jpg', 'coco_url': '.jpg',
#           'height': 480, 'width': 640, 'date_captured': '2013-11',
#           'flickr_url': 'http://farm4.staticflickr.com/.jpg', 'id': 57870},
#           {},{}]
# licenses = [{'url': 'y-nc-sa/2.0/', 'id': 1, 'name': 'Attr License'},
#           {},{}]
# annotations = [{'image_id': 330756, 'id': 816783, 'caption': 'a pair of scissors'},
#           {},{}]

# captions_val2014.json
# dict_keys(['info', 'images', 'licenses', 'annotations'])
# info = {'description': 'COCO Dataset', 'url': '.org', 'version': '1.0',
#           'year': 2014, 'contributor': 'COCO Consortium', 'date_created': '2017'}
# images = {'license': 3, 'file_name': '.jpg', 'coco_url': '.jpg',
#           'height': 343, 'width': 500, 'date_captured': '2013-11-17 00:25:38',
#           'flickr_url': 'http://farm4.staticflickr.com/.jpg', 'id': 16931},
#            {},{}]
# licenses = [{'url': 'c-sa/2.0/', 'id': 1, 'name': 'AttributeAlike License'},
#            {},{}]
# annotations = [{'image_id': 548246, 'id': 726786, 'caption': 'A group of men talk on a tenn'},
#            {},{}]
with open(filename, 'r', encoding='utf-8') as file:
    data = json.load(file)   # type(data) = dict
    # 字典是支持嵌套的，
    print(data['annotations'])
    #for i,j in enumerate(data):
    #    print(i)
    #    print(j)


