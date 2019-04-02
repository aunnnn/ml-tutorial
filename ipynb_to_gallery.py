"""Convert jupyter notebook to sphinx gallery notebook styled examples.

Usage: python ipynb_to_gallery.py "./blog_content_source/**/*.ipynb"

Dependencies:
pypandoc: install using `pip install pypandoc`

From https://gist.github.com/chsasank/7218ca16f8d022e02a9c0deb94a310fe
"""
import pypandoc as pdoc
import json
import glob

extra_args = ['-fmarkdown-implicit_figures']

def convert_ipynb_to_gallery(file_name):
    python_file = ""

    nb_dict = json.load(open(file_name))
    cells = nb_dict['cells']

    for i, cell in enumerate(cells):
        if i == 0:  
            assert cell['cell_type'] == 'markdown', \
                'First cell has to be markdown'

            md_source = ''.join(cell['source'])
            rst_source = pdoc.convert_text(md_source, 'rst', 'md', extra_args=extra_args)
            python_file = '"""\n' + rst_source + '\n"""'
        else:
            if cell['cell_type'] == 'markdown':
                md_source = ''.join(cell['source'])
                rst_source = pdoc.convert_text(md_source, 'rst', 'md', extra_args=extra_args)
                commented_source = '\n'.join(['# ' + x for x in
                                              rst_source.split('\n')])
                python_file = python_file + '\n\n\n' + '#' * 70 + '\n' + \
                    commented_source
            elif cell['cell_type'] == 'code':
                source = ''.join(cell['source'])
                python_file = python_file + '\n' * 2 + source

    python_file = python_file.replace("\n%", "\n# %")
    open(file_name.replace('.ipynb', '.py'), 'w').write(python_file)

if __name__ == '__main__':
    import sys
    for f in glob.glob(sys.argv[-1], recursive=True):
        print("Working on", f)
        convert_ipynb_to_gallery(f)
    # convert_ipynb_to_gallery(sys.argv[-1]