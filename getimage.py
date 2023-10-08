import datetime
import os
import matplotlib.pyplot as plt
import numpy as np
# The following is not a package. It is a file utils.py which should be in the same folder as this notebook.
from utils import plot_image
from sentinelhub import (
    CRS,
    BBox,
    DataCollection,
    DownloadRequest,
    MimeType,
    MosaickingOrder,
    SentinelHubDownloadClient,
    SentinelHubRequest,
    bbox_to_dimensions,
)

from sentinelhub import SHConfig
config = SHConfig()
config.sh_client_id = "31b89ac3-f6d9-4f20-b0a7-e5378703d2d9"
config.sh_client_secret = "Rdv|EbcB:hLOh&J0u5I;^_)kxrW}C2N&>8;V@Cs2"
if not config.sh_client_id or not config.sh_client_secret:
    print("Warning! To use Process API, please provide the credentials (OAuth client ID and client secret).")
config.save()


evalscript_true_color = """
        //VERSION=3

        function setup() {
            return {
                input: [{
                    bands: ["B02", "B03", "B04"]
                }],
                output: {
                    bands: 3
                }
            };
        }

        function evaluatePixel(sample) {
            return [sample.B04, sample.B03, sample.B02];
        }
"""


def get_true_color_request(time_interval, betsiboka_bbox, betsiboka_size):
    return SentinelHubRequest(
        evalscript=evalscript_true_color,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L1C,
                time_interval=time_interval,
                mosaicking_order=MosaickingOrder.LEAST_CC,
            )
        ],
        responses=[SentinelHubRequest.output_response(
            "default", MimeType.PNG)],
        bbox=betsiboka_bbox,
        size=betsiboka_size,
        config=config,
    )


def getImagesByCoordinatesAndImages(bboxdimensions, start, end):
    resolution = 60
    betsiboka_bbox = BBox(bbox=bboxdimensions, crs=CRS.WGS84)
    betsiboka_size = bbox_to_dimensions(betsiboka_bbox, resolution=resolution)
    if betsiboka_size[0] > 2500:
        return ("knowing that each pixel is 60 meter, You can't generate with an image of more than 2500 pixel in the height")
    if betsiboka_size[1] > 2500:
        return ("knowing that each pixel is 60 meter, You can't generate with an image of more than 2500 pixel in the width")

    n_chunks = 13
    tdelta = (end - start) / n_chunks
    edges = [(start + i * tdelta).date().isoformat() for i in range(n_chunks)]
    slots = [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]

    # create a list of requests
    list_of_requests = [get_true_color_request(
        slot, betsiboka_bbox, betsiboka_size) for slot in slots]
    list_of_requests = [request.download_list[0]
                        for request in list_of_requests]

    # download data with multiple threads
    data = SentinelHubDownloadClient(config=config).download(
        list_of_requests, max_threads=5)
    return data
