#!/usr/bin/env python
# coding: utf-8

from __future__ import annotations

import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

def show(frame: np.ndarray) -> None:
    cv2.imshow("test", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def plot_point(frame: np.ndarray, x: int, y: int, color: tuple[int, int, int] = (0,255,0), radius: int = 0) -> np.ndarray:
    thickness = -1
    return cv2.circle(frame, (x,y), radius, color, thickness)

def plot_line(frame: np.ndarray, p1: np.ndarray | list[int], p2: np.ndarray | list[int], color: tuple[int, int, int] = (0,191,255), thickness: int = 1) -> np.ndarray:
    return cv2.line(frame, (p1[0],p1[1]), (p2[0],p2[1]), color, thickness)

def plot_raw_curve(array: list[float]) -> None:
    plt.clf()
    plt.plot(array,label = 'Raw Data',alpha=0.7)

def plot_smoothed_curve(array: list[float] | np.ndarray, label: str) -> None:
    plt.plot(array,label = label,alpha=0.7)

def mark_valley(index: int, value: float) -> None:
    plt.scatter(index,value,color='green')

def mark_peak(index: int, value: float) -> None:
    plt.scatter(index,value,color='red')

def save_ratio_plot(plot_dir: str, filename: str) -> None:
    plt.savefig(os.path.join(plot_dir,filename[:-3]+'png'))
    plt.clf()
