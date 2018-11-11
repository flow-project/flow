"""
A preliminary attempt at parsing an RST file's math syntax
in order to make math render as inline rather than display
mode. This doesn't work as of yet but might be useful.

It could, however, be not useful if there's a pandoc option
for converting .md to .rst that makes math inline and not
display. Keeping it around, though.
"""

import re

s = """Define

.. math:: v_{des}

as the desired velocity,

.. math:: 1^k

 a vector of ones of length"""

with open('/Users/nishant/Downloads/tutorialtest.rst', 'r') as myfile:
    s = myfile.read()

print([elem[11:-2] for elem in re.findall('\n.. math:: *\S*\n\n', s)])  # noqa
