import urllib2
import urllib
import json
import numpy as np
import cv2
import untangle
import scipy.misc
from google.cloud import storage
import tempfile
from preprocess import *

def run(tag_classes, start_count):

    temp = tempfile.NamedTemporaryFile()
    client = storage.Client()
    bucket = client.get_bucket("dcgan-161707-mlengine")

    count = int(start_count)
    maxsize = 512

    # add glasses + hair color / combos
    #                     1            2            3
    # tag_classes = ["blue_hair", "red_hair", "blonde_hair"]
    imgs = []
    tags = []
    tagname = []

    for idx, tag in enumerate(tag_classes):
        print("Now downloading... " + tag)
        for i in xrange(10):
            stringreturn = urllib2.urlopen("http://safebooru.org/index.php?page=dapi&s=post&q=index&tags=1girl+"+tag+"&pid="+str(i+20)).read()
            xmlreturn = untangle.parse(stringreturn)
            for post in xmlreturn.posts.post:
                imgurl = "http:" + post["sample_url"]
                print imgurl
                if ("png" in imgurl) or ("jpg" in imgurl):
                    resp = urllib.urlopen(imgurl)
                    image = np.asarray(bytearray(resp.read()), dtype="uint8")
                    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

                    count += 1
                    temp.seek(0,0)
                    scipy.misc.imsave(temp,image, "jpeg")
                    blob = bucket.blob("imgs/"+str(count)+".jpg")
                    temp.seek(0,0)
                    if process_img(temp.name) != 0:
                        blob.upload_from_file(temp,content_type='image/jpeg')
                        imgs.append("imgs/"+str(count)+".jpg")
                        tags.append(idx)
                        tagname.append(tag)

if __name__ == '__main__':
    if len(sys.argv) >= 2:
        run(eval(sys.argv[1]), sys.argv[2])
    # py download.py '["blue_hair"]' 0
