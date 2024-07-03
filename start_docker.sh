#!/bin/bash

sudo docker run -v /home/joey/:/home/joey/ -it --rm --runtime nvidia --network host my_updated_docker:v00



