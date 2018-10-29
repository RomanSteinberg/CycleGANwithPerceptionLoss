import dominate
from dominate.tags import *
import os


class HTML:
    def __init__(self, web_dir, title, reflesh=0):
        self.title = title
        self.web_dir = web_dir
        self.img_dir = os.path.join(self.web_dir, 'images')
        if not os.path.exists(self.web_dir):
            os.makedirs(self.web_dir)
        if not os.path.exists(self.img_dir):
            os.makedirs(self.img_dir)
        # print(self.img_dir)

        self.doc = dominate.document(title=title)
        if reflesh > 0:
            with self.doc.head:
                meta(http_equiv="reflesh", content=str(reflesh))

        self.top_div = div(style="display:block")
        self.doc.add(self.top_div)

    def get_image_dir(self):
        return self.img_dir

    def add_url(self, url, text):
        if url is not None:
            with self.top_div:
                with a(href=url, style="padding-right:30px"):
                    h2(text, style="display:inline-block")

    def add_header(self, str):
        with self.doc:
            h3(str)

    def add_table(self, border=1):
        self.t = table(border=border, style="table-layout: fixed;")
        self.doc.add(self.t)

    def add_images(self, img_rel_paths, txts, width=400):
        self.add_table()
        with self.t:
            with tr():
                for im_rel_path, txt in zip(img_rel_paths, txts):
                    with td(style="word-wrap: break-word;", halign="center", valign="top"):
                        with p():
                            with a(href=im_rel_path):
                                img(style="width:%dpx" % width, src=im_rel_path)
                            br()
                            p(txt)

    def save(self):
        html_file = '%s/index.html' % self.web_dir
        f = open(html_file, 'wt')
        f.write(self.doc.render())
        f.close()


if __name__ == '__main__':
    html = HTML('web/', 'test_html')
    html.add_header('hello world')

    ims = []
    txts = []
    for n in range(4):
        ims.append('image_%d.png' % n)
        txts.append('text_%d' % n)
    html.add_images(ims, txts)
    html.save()
