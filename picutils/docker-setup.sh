#!/bin/bash

python3 -c 'import os;items=[x for x in os.listdir("template_files") if x.endswith("Dockerfile")];
if len(items)>1:
  print("Choose one of the Dockerfiles:")
  for i,t in enumerate(items):
    print("{}: {}".format(i+1,t))'

ITEM=`python3 -c 'import os;items=[x for x in os.listdir("template_files") if x.endswith("Dockerfile")];
if len(items)>1:
  choice=int(input(""))
else:
  choice=1
item = items[choice - 1]
print(item)'`
echo $ITEM

read -p "Should I add the link directory to the runtime? " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    LINK_FLAG="yes"
else
	LINK_FLAG=""
fi

# accumulate proper files in catalogues
rm -rf /tmp/pictec-build || echo "Nothing to purge"
mkdir -p /tmp/pictec-build
cp -r template_files/* /tmp/pictec-build
if [ -z "$LINK_FLAG" ]; then
	echo "Not linking"
else
	cp -r link/* /tmp/pictec-build || echo "No linkable folder for building exists"
fi
mv /tmp/pictec-build/$ITEM /tmp/pictec-build/Dockerfile
cp -r Q /tmp/pictec-build

# build a docker image from prepared path
sudo docker build -t pictec-workspace /tmp/pictec-build
