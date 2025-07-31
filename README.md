# 🚗 RiskRadarSystem - Sistema de Detección de Riesgo 3D con ROI

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.5%2B-orange)
![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-red)
![YOLOv8](https://img.shields.io/badge/YOLO-v8-green)
![License](https://img.shields.io/badge/license-MIT-green)

Sistema avanzado de detección de riesgo en tiempo real que combina detección de objetos 3D con YOLOv8, estimación de profundidad con MiDaS y análisis de riesgo en una Región de Interés (ROI) optimizada.

## 🚀 Características principales

- 🔍 **Detección de objetos** con YOLOv8 (modelos personalizados y COCO)
- 📏 **Estimación de profundidad 3D** usando MiDaS de Intel
- 🎯 **Procesamiento optimizado** con Región de Interés (ROI)
- 🌡️ **Mapa de calor de riesgo** dinámico con decaimiento
- 📊 **Análisis estadístico** completo con visualizaciones
- ⚡ **Seguimiento de objetos** con OpenCV Tracker (CSRT)
- 📹 **Procesamiento de video** con salida anotada

## 📦 Dependencias

```bash
pip install torch torchvision opencv-python ultralytics numpy matplotlib
