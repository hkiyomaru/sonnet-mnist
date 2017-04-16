#!/bin/sh

echo 'wget http://mattmahoney.net/dc/text8.zip -O ../data/text8.zip'
wget http://mattmahoney.net/dc/text8.zip -O ../data/text8.zip
echo 'unzip ../data/text8.zip'
unzip ../data/text8.zip -d ../data/
