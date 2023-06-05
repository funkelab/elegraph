for filename in $(ls /groups/funke/home/tame/seamcellcoordinates)
do 
    python my_script.py $filename
done