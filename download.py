import urllib2
import urllib
import json
import numpy as np
import cv2
import untangle
import scipy.misc
from google.cloud import storage
import tempfile
from redo import retry
from preprocess import *

@retry(urllib2.URLError, tries=4, delay=3, backoff=2)
def readURL(tag, i):
    return urllib2.urlopen("http://safebooru.org/index.php?page=dapi&s=post&q=index&tags=1girl+"+tag+"&pid="+str(i+20)).read()

def run(tag_classes, start_count):

    if not os.path.exists(tag_classes[0]):
        os.makedirs(tag_classes[0])
    temp = tempfile.NamedTemporaryFile(dir=tag_classes[0])
    client = storage.Client()
    bucket = client.get_bucket("bucket_name_here")

    count = int(start_count)
    fail = 0
    maxsize = 512

    # add glasses + hair color / combos
    #                     0            1            2              3
    # tag_classes = ["blue_hair", "red_hair", "blonde_hair", "black_hair"]
    # ENDS UP MAKING ~1000 IMAGES PER CLASS, CHANGE XRANGE FOR MORE IMAGES

    imgs = []
    tags = []
    tagname = []

    for idx, tag in enumerate(tag_classes):
        print("Now downloading... " + tag)
        for i in xrange(10):
            stringreturn = readURL(tag, i)
            xmlreturn = untangle.parse(stringreturn)
            for post in xmlreturn.posts.post:
                imgurl = "http:" + post["sample_url"]
                print imgurl
                if ("png" in imgurl) or ("jpg" in imgurl):
                    resp = urllib.urlopen(imgurl)
                    image = np.asarray(bytearray(resp.read()), dtype="uint8")
                    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

                    temp.seek(0,0)
                    scipy.misc.imsave(temp,image[...,::-1], "jpeg")
                    temp.seek(0,0)
                    if process_img(temp.name) != 0:
                        count += 1
                        blob = bucket.blob("imgs/"+str(count)+".jpg")
                        blob.upload_from_file(temp,content_type='image/jpeg')
                        imgs.append("imgs/"+str(count)+".jpg")
                        tags.append(idx)
                        tagname.append(tag)
                    else:
                        fail += 1
    print("Done: %d failed" % fail)

if __name__ == '__main__':
    if len(sys.argv) >= 2:
        run(eval(sys.argv[1]), sys.argv[2])
    # py download.py '["blue_hair"]' 0
