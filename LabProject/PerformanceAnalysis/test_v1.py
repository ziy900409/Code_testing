# -*- coding: utf-8 -*-
"""
Created on Fri Aug  8 14:31:49 2025

@author: Hsin.YH.Yang
"""
file_path = r"D:\BenQ_Project\01_UR_lab\00_BQE\2025_06 Lab Opening\motion\S1\S1_Post_Spider30_EC.c3d"
import ezc3d
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from pathlib import Path
from ..models.c3d_models import (
    C3DFileInfo, C3DParameters, MarkerInfo, AnalogChannelInfo,
    PointData, AnalogData, C3DData, FrameData
)
from ..core.exceptions import FileProcessingError

logger = logging.getLogger(__name__)

class C3DParser:
    """C3D 檔案解析器"""
    
    def __init__(self):
        self.supported_versions = ["1.1", "2.0"]
    
    def parse_file(self, file_path: Path, file_id: str, user_id: str) -> C3DData:
        """
        解析 C3D 檔案
        
        Args:
            file_path: 檔案路徑
            file_id: 檔案 ID
            user_id: 用戶 ID
            
        Returns:
            C3DData: 解析後的 C3D 數據
        """
        try:
            logger.info(f"Starting C3D parsing: {file_path}")
            
            # 使用 ezc3d 讀取檔案
            c3d_reader = ezc3d.c3d(str(file_path))
            
            # 提取基本資訊
            file_info = self._extract_file_info(c3d_reader, file_path, file_id, user_id)
            
            # 提取參數
            parameters = self._extract_parameters(c3d_reader)
            
            # 提取標記點資訊
            markers = self._extract_marker_info(c3d_reader)
            
            # 提取類比通道資訊
            analog_channels = self._extract_analog_channel_info(c3d_reader)
            
            # 提取點數據
            point_data = self._extract_point_data(c3d_reader, markers)
            
            # 提取類比數據
            analog_data = self._extract_analog_data(c3d_reader, analog_channels)
            
            # 創建完整的 C3D 數據對象
            c3d_data = C3DData(
                file_info=file_info,
                parameters=parameters,
                markers=markers,
                analog_channels=analog_channels,
                point_data=point_data,
                analog_data=analog_data
            )
            
            logger.info(f"C3D parsing completed successfully: {file_id}")
            return c3d_data
            
        except Exception as e:
            error_msg = f"Failed to parse C3D file {file_path}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise FileProcessingError(error_msg)
    
    def _extract_file_info(self, c3d_reader: ezc3d.c3d, file_path: Path, file_id: str, user_id: str) -> C3DFileInfo:
        """提取檔案基本資訊"""
        try:
            # 獲取參數
            params = c3d_reader['parameters']
            
            # 提取基本參數
            point_rate = float(params['POINT']['RATE']['value'][0])
            first_frame = int(params['POINT']['FRAMES']['value'][0])
            last_frame = int(params['POINT']['FRAMES']['value'][1])
            total_frames = last_frame - first_frame + 1
            
            # 獲取標記點數量
            n_markers = len(params['POINT']['LABELS']['value'])
            
            # 獲取類比通道數量
            n_analog_channels = 0
            if 'ANALOG' in params and 'LABELS' in params['ANALOG']:
                n_analog_channels = len(params['ANALOG']['LABELS']['value'])
            
            # 計算持續時間
            duration = total_frames / point_rate
            
            return C3DFileInfo(
                file_id=file_id,
                filename=file_path.name,
                file_size=file_path.stat().st_size,
                upload_time=datetime.utcnow(),
                frame_rate=point_rate,
                total_frames=total_frames,
                n_markers=n_markers,
                n_analog_channels=n_analog_channels,
                duration=duration,
                user_id=user_id
            )
            
        except Exception as e:
            raise FileProcessingError(f"Failed to extract file info: {str(e)}")
    
    def _extract_parameters(self, c3d_reader: ezc3d.c3d) -> C3DParameters:
        """提取 C3D 參數"""
        try:
            params = c3d_reader['parameters']
            
            # 提取點參數
            point_params = params['POINT']
            point_rate = float(point_params['RATE']['value'][0])
            first_frame = int(point_params['FRAMES']['value'][0])
            last_frame = int(point_params['FRAMES']['value'][1])
            scale_factor = float(point_params['SCALE']['value'][0])
            data_start = int(point_params['DATA_START']['value'][0])
            
            # 提取類比參數
            analog_rate = point_rate
            analog_samples_per_frame = 1
            if 'ANALOG' in params and 'RATE' in params['ANALOG']:
                analog_rate = float(params['ANALOG']['RATE']['value'][0])
                analog_samples_per_frame = int(analog_rate / point_rate)
            
            # 提取標籤和描述
            labels = [label.strip() for label in point_params['LABELS']['value']]
            descriptions = []
            if 'DESCRIPTIONS' in point_params:
                descriptions = [desc.strip() for desc in point_params['DESCRIPTIONS']['value']]
            
            # 提取單位
            units = []
            if 'UNITS' in point_params:
                units = [unit.strip() for unit in point_params['UNITS']['value']]
            
            # 提取最大插值間隙
            max_interpolation_gap = 0
            if 'MAX_INTERPOLATION_GAP' in point_params:
                max_interpolation_gap = int(point_params['MAX_INTERPOLATION_GAP']['value'][0])
            
            return C3DParameters(
                point_rate=point_rate,
                analog_rate=analog_rate,
                first_frame=first_frame,
                last_frame=last_frame,
                max_interpolation_gap=max_interpolation_gap,
                scale_factor=scale_factor,
                data_start=data_start,
                analog_samples_per_frame=analog_samples_per_frame,
                labels=labels,
                descriptions=descriptions,
                units=units
            )
            
        except Exception as e:
            raise FileProcessingError(f"Failed to extract parameters: {str(e)}")
    
    def _extract_marker_info(self, c3d_reader: ezc3d.c3d) -> List[MarkerInfo]:
        """提取標記點資訊"""
        try:
            params = c3d_reader['parameters']['POINT']
            labels = [label.strip() for label in params['LABELS']['value']]
            descriptions = []
            
            if 'DESCRIPTIONS' in params:
                descriptions = [desc.strip() for desc in params['DESCRIPTIONS']['value']]
            
            markers = []
            for i, label in enumerate(labels):
                marker = MarkerInfo(
                    name=label,
                    index=i,
                    is_valid=True,
                    description=descriptions[i] if i < len(descriptions) else None
                )
                markers.append(marker)
            
            return markers
            
        except Exception as e:
            raise FileProcessingError(f"Failed to extract marker info: {str(e)}")
    
    def _extract_analog_channel_info(self, c3d_reader: ezc3d.c3d) -> List[AnalogChannelInfo]:
        """提取類比通道資訊"""
        try:
            channels = []
            params = c3d_reader['parameters']
            
            if 'ANALOG' not in params:
                return channels
            
            analog_params = params['ANALOG']
            
            if 'LABELS' not in analog_params:
                return channels
            
            labels = [label.strip() for label in analog_params['LABELS']['value']]
            
            # 提取單位
            units = []
            if 'UNITS' in analog_params:
                units = [unit.strip() for unit in analog_params['UNITS']['value']]
            
            # 提取縮放因子
            scales = []
            if 'SCALE' in analog_params:
                scales = analog_params['SCALE']['value']
            
            # 提取偏移量
            offsets = []
            if 'OFFSET' in analog_params:
                offsets = analog_params['OFFSET']['value']
            
            # 提取描述
            descriptions = []
            if 'DESCRIPTIONS' in analog_params:
                descriptions = [desc.strip() for desc in analog_params['DESCRIPTIONS']['value']]
            
            for i, label in enumerate(labels):
                channel = AnalogChannelInfo(
                    name=label,
                    index=i,
                    unit=units[i] if i < len(units) else "",
                    scale=float(scales[i]) if i < len(scales) else 1.0,
                    offset=float(offsets[i]) if i < len(offsets) else 0.0,
                    description=descriptions[i] if i < len(descriptions) else None
                )
                channels.append(channel)
            
            return channels
            
        except Exception as e:
            raise FileProcessingError(f"Failed to extract analog channel info: {str(e)}")
    
    def _extract_point_data(self, c3d_reader: ezc3d.c3d, markers: List[MarkerInfo]) -> List[PointData]:
        """提取點數據"""
        try:
            # 獲取點數據 - 形狀為 (4, n_markers, n_frames)
            points_raw = c3d_reader['data']['points']
            n_markers, n_frames = points_raw.shape[1], points_raw.shape[2]
            
            point_data_list = []
            
            for marker_idx, marker in enumerate(markers):
                if marker_idx >= n_markers:
                    break
                
                # 提取該標記點的所有幀數據
                marker_data = points_raw[:, marker_idx, :]  # (4, n_frames)
                
                # 轉換為列表格式 [frame][x,y,z,residual]
                coordinates = []
                valid_frames = []
                
                for frame_idx in range(n_frames):
                    x, y, z, residual = marker_data[:, frame_idx]
                    coordinates.append([float(x), float(y), float(z), float(residual)])
                    
                    # 檢查點是否有效（residual >= 0 表示有效）
                    valid_frames.append(residual >= 0)
                
                point_data = PointData(
                    marker_name=marker.name,
                    coordinates=coordinates,
                    valid_frames=valid_frames
                )
                point_data_list.append(point_data)
            
            return point_data_list
            
        except Exception as e:
            raise FileProcessingError(f"Failed to extract point data: {str(e)}")
    
    def _extract_analog_data(self, c3d_reader: ezc3d.c3d, channels: List[AnalogChannelInfo]) -> List[AnalogData]:
        """提取類比數據"""
        try:
            analog_data_list = []
            
            if not channels:
                return analog_data_list
            
            # 獲取類比數據
            if 'analogs' not in c3d_reader['data']:
                return analog_data_list
            
            analogs_raw = c3d_reader['data']['analogs']  # (n_channels, n_samples)
            
            # 獲取採樣率
            params = c3d_reader['parameters']
            analog_rate = float(params['POINT']['RATE']['value'][0])
            if 'ANALOG' in params and 'RATE' in params['ANALOG']:
                analog_rate = float(params['ANALOG']['RATE']['value'][0])
            
            for channel_idx, channel in enumerate(channels):
                if channel_idx >= analogs_raw.shape[0]:
                    break
                
                # 提取該通道的數據
                channel_values = analogs_raw[channel_idx, :].tolist()
                
                # 應用縮放因子和偏移量
                scaled_values = [
                    float(val * channel.scale + channel.offset)
                    for val in channel_values
                ]
                
                analog_data = AnalogData(
                    channel_name=channel.name,
                    values=scaled_values,
                    sample_rate=analog_rate,
                    unit=channel.unit
                )
                analog_data_list.append(analog_data)
            
            return analog_data_list
            
        except Exception as e:
            raise FileProcessingError(f"Failed to extract analog data: {str(e)}")
    
    def extract_frame_data(self, c3d_data: C3DData, frame_number: int) -> FrameData:
        """
        提取指定幀的數據
        
        Args:
            c3d_data: C3D 數據對象
            frame_number: 幀號碼 (0-based)
            
        Returns:
            FrameData: 該幀的數據
        """
        try:
            if frame_number < 0 or frame_number >= c3d_data.file_info.total_frames:
                raise ValueError(f"Frame number {frame_number} out of range")
            
            # 計算時間戳
            timestamp = frame_number / c3d_data.file_info.frame_rate
            
            # 提取標記點數據
            markers_data = {}
            for point_data in c3d_data.point_data:
                if frame_number < len(point_data.coordinates):
                    markers_data[point_data.marker_name] = point_data.coordinates[frame_number]
            
            # 提取類比數據（需要計算對應的採樣點）
            analogs_data = {}
            if c3d_data.analog_data:
                analog_rate = c3d_data.parameters.analog_rate
                point_rate = c3d_data.parameters.point_rate
                samples_per_frame = int(analog_rate / point_rate)
                
                for analog_data in c3d_data.analog_data:
                    start_idx = frame_number * samples_per_frame
                    end_idx = start_idx + samples_per_frame
                    
                    if start_idx < len(analog_data.values):
                        # 取該幀的平均值
                        frame_values = analog_data.values[start_idx:end_idx]
                        if frame_values:
                            analogs_data[analog_data.channel_name] = sum(frame_values) / len(frame_values)
            
            return FrameData(
                frame_number=frame_number,
                timestamp=timestamp,
                markers=markers_data,
                analogs=analogs_data
            )
            
        except Exception as e:
            raise FileProcessingError(f"Failed to extract frame data: {str(e)}")
    
    def validate_c3d_file(self, file_path: Path) -> Tuple[bool, str]:
        """
        驗證 C3D 檔案
        
        Args:
            file_path: 檔案路徑
            
        Returns:
            Tuple[bool, str]: (是否有效, 錯誤訊息)
        """
        try:
            # 檢查檔案是否存在
            if not file_path.exists():
                return False, "File does not exist"
            
            # 檢查檔案大小
            if file_path.stat().st_size == 0:
                return False, "File is empty"
            
            # 嘗試用 ezc3d 讀取檔案
            c3d_reader = ezc3d.c3d(str(file_path))
            
            # 檢查是否有必要的參數
            if 'parameters' not in c3d_reader:
                return False, "Invalid C3D file: missing parameters"
            
            params = c3d_reader['parameters']
            if 'POINT' not in params:
                return False, "Invalid C3D file: missing POINT parameters"
            
            # 檢查是否有數據
            if 'data' not in c3d_reader:
                return False, "Invalid C3D file: missing data section"
            
            if 'points' not in c3d_reader['data']:
                return False, "Invalid C3D file: missing point data"
            
            return True, "Valid C3D file"
            
        except Exception as e:
            return False, f"Invalid C3D file: {str(e)}"

# 修復 datetime import
from datetime import datetime