#!/usr/bin/env python3

"""Generates a matrix for docker releases through github actions

Will output a condensed version of the matrix. Will include fllowing:
    * CUDA version short
    * CUDA full version
    * CUDNN version short
    * Image type either runtime or devel
    * Platform linux/arm64,linux/amd64

"""

import json

import generate_binary_build_matrix


DOCKER_IMAGE_TYPES = ["runtime", "devel"]
DOCKER_PLATFORMS = {
    "amd64": {
        "platform": "linux/amd64",
        "runner": "linux.2xlarge",
    },
    "arm64": {
        "platform": "linux/arm64",
        "runner": "linux.arm64.m7g.4xlarge",
    },
}


def generate_docker_matrix() -> dict[str, list[dict[str, str]]]:
    ret: list[dict[str, str]] = []
    # CUDA Docker images are built natively for amd64 and arm64 as both runtime
    # and devel. CPU arm64 image is only available as runtime.
    for cuda, version in generate_binary_build_matrix.CUDA_ARCHES_FULL_VERSION.items():
        for image in DOCKER_IMAGE_TYPES:
            for arch, platform_config in DOCKER_PLATFORMS.items():
                ret.append(
                    {
                        "cuda": cuda,
                        "cuda_full_version": version,
                        "cudnn_version": generate_binary_build_matrix.CUDA_ARCHES_CUDNN_VERSION[
                            cuda
                        ],
                        "image_type": image,
                        "platform": platform_config["platform"],
                        "arch": arch,
                        "runner": platform_config["runner"],
                    }
                )
    ret.append(
        {
            "cuda": "cpu",
            "cuda_full_version": "",
            "cudnn_version": "",
            "image_type": "runtime",
            "platform": "linux/arm64",
            "arch": "arm64",
            "runner": DOCKER_PLATFORMS["arm64"]["runner"],
        }
    )

    return {"include": ret}


if __name__ == "__main__":
    build_matrix = generate_docker_matrix()
    print(json.dumps(build_matrix))
