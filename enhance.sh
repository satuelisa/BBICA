#!/bin/zsh
loc="/Users/elisa/Dropbox/Research/Topics/Arboles/CA/RAW/jul25b/images"
for file in `ls -1 ${loc}/*.JPG`; do
    frame=`basename $file .JPG`
    echo Enhancing $frame
    convert $file -separate -normalize -combine ${loc}/enhanced/normalized/$frame.png
    convert -equalize ${loc}/enhanced/normalized/$frame.png ${loc}/enhanced/equalized/$frame.png
    # the script redist is from http://www.fmwconcepts.com/imagemagick/redist/n
    redist -s uniform ${loc}/enhanced/equalized/$frame.png ${loc}/enhanced/uniform/$frame.png
    # 50,220 ground too red, whites turn blues
    # same with 80,180
    convert ${loc}/enhanced/uniform/$frame.png -modulate 100,150 ${loc}/enhanced/modulated/$frame.png 
    convert -transparent black ${loc}/enhanced/modulated/$frame.png ${loc}/enhanced/$frame.png
done
