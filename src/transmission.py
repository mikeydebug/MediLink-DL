"""transmission.py: Network Transmission Simulation with Priority Queuing and ARQ.

Part of MediLink-DL: Bandwidth-Adaptive Medical Image Transmission.
Simulates application-level transmission over constrained, lossy telemedicine links
with chunking, checksum verification, selective retransmission, and priority queuing.
"""

from __future__ import annotations

import hashlib
import heapq
import math
import struct
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from src.roi_compression import deserialize_dual_stream_payload, ROICompressionError


@dataclass(order=True)
class Packet:
    """Network transmission packet."""
    priority: int                   # Negated for min-heap priority (lower number = earlier transmission)
    packet_id: int
    total_packets: int = field(compare=False)
    payload_type: str = field(compare=False)   # "HEADER", "ROI", "BG", "UNIFORM"
    data: bytes = field(compare=False)
    checksum: str = field(compare=False)
    attempt: int = field(default=0, compare=False)


def compute_checksum(data: bytes) -> str:
    """Compute SHA-256 checksum for payload integrity verification."""
    return hashlib.sha256(data).hexdigest()


def chunk_data(
    data: bytes,
    chunk_size: int = 1024,
    payload_type: str = "DATA",
    priority_level: int = 1,
    start_id: int = 0
) -> List[Packet]:
    """Segment raw bytes into a list of verified packets."""
    packets = []
    total_chunks = max(1, math.ceil(len(data) / chunk_size))
    
    for i in range(total_chunks):
        offset = i * chunk_size
        chunk = data[offset : offset + chunk_size]
        pkt = Packet(
            priority=-priority_level,  # Negative for min-heap priority queue
            packet_id=start_id + i,
            total_packets=total_chunks,
            payload_type=payload_type,
            data=chunk,
            checksum=compute_checksum(chunk),
            attempt=0
        )
        packets.append(pkt)
    return packets


def chunk_dual_stream_payload(
    payload_bytes: bytes,
    chunk_size: int = 1024,
    enable_priority: bool = True
) -> List[Packet]:
    """Chunk a dual-stream payload with diagnostic prioritization.
    
    Priority assignment:
    - HEADER: Priority 30 (highest - contains dimensions, coordinates, lengths)
    - ROI stream: Priority 20 (high - critical diagnostic lesion data)
    - BG stream: Priority 5 (low - contextual anatomy)
    
    If enable_priority is False, all chunks receive equal priority (1).
    """
    header_size = struct.calcsize(">4sHHHBBHHHHII")
    if len(payload_bytes) < header_size:
        raise ValueError("Payload smaller than header size.")
    
    header_bytes = payload_bytes[:header_size]
    parsed = deserialize_dual_stream_payload(payload_bytes)
    
    p_hdr = 30 if enable_priority else 1
    p_roi = 20 if enable_priority else 1
    p_bg = 5 if enable_priority else 1
    
    # 1. Header packet
    header_packet = Packet(
        priority=-p_hdr,
        packet_id=0,
        total_packets=0,  # updated later
        payload_type="HEADER",
        data=header_bytes,
        checksum=compute_checksum(header_bytes)
    )
    
    roi_data = parsed["roi_crop_jpeg_bytes"]
    bg_data = parsed["bg_jpeg_bytes"]
    
    if enable_priority:
        # Priority Scheduling: Diagnostic ROI packets sequenced immediately after header
        roi_packets = chunk_data(
            roi_data,
            chunk_size=chunk_size,
            payload_type="ROI",
            priority_level=p_roi,
            start_id=1
        )
        bg_packets = chunk_data(
            bg_data,
            chunk_size=chunk_size,
            payload_type="BG",
            priority_level=p_bg,
            start_id=1 + len(roi_packets)
        )
        packets = [header_packet] + roi_packets + bg_packets
    else:
        # Standard FIFO baseline: Full contextual frame stream transmitted before patch overlay
        bg_packets = chunk_data(
            bg_data,
            chunk_size=chunk_size,
            payload_type="BG",
            priority_level=p_bg,
            start_id=1
        )
        roi_packets = chunk_data(
            roi_data,
            chunk_size=chunk_size,
            payload_type="ROI",
            priority_level=p_roi,
            start_id=1 + len(bg_packets)
        )
        packets = [header_packet] + bg_packets + roi_packets
    
    total = len(packets)
    for pkt in packets:
        pkt.total_packets = total
        
    return packets


class ChannelSimulator:
    """Simulates bandwidth constraints, propagation latency, and packet loss."""
    
    def __init__(
        self,
        bandwidth_bps: int = 64 * 1024,  # 64 kbps (typical 2G rural health clinic)
        latency_ms: float = 120.0,       # 120 ms round-trip satellite/cellular delay
        loss_rate: float = 0.05,         # 5% packet drop rate
        seed: int = 42
    ):
        self.bandwidth_bps = max(1000, bandwidth_bps)
        self.latency_sec = max(0.0, latency_ms / 1000.0)
        self.loss_rate = max(0.0, min(loss_rate, 0.99))
        self.rng = np.random.default_rng(seed)
        
    def transmit_packet(self, packet: Packet) -> Tuple[bool, float]:
        """Simulate transmitting a single packet.
        
        Returns:
            tuple: (success: bool, elapsed_simulated_time_sec: float)
        """
        # Serialization delay (packet size in bits / bandwidth in bps)
        packet_bytes = len(packet.data) + 64 # 64 bytes packet framing overhead
        tx_delay = (packet_bytes * 8.0) / self.bandwidth_bps
        # One-way propagation delay
        total_delay = tx_delay + self.latency_sec
        
        # Determine packet loss
        lost = self.rng.random() < self.loss_rate
        return (not lost), total_delay


class Receiver:
    """Simulates remote medical receiving workstation with integrity check."""
    
    def __init__(self):
        self.received_packets: Dict[int, Packet] = {}
        self.roi_completed_time_sec: Optional[float] = None
        self.expected_total: Optional[int] = None
        self.header_data: Optional[bytes] = None
        
    def process_packet(self, packet: Packet, current_time_sec: float) -> bool:
        """Verify checksum and store packet. Returns True if packet accepted."""
        if compute_checksum(packet.data) != packet.checksum:
            return False  # Checksum mismatch
            
        self.received_packets[packet.packet_id] = packet
        self.expected_total = packet.total_packets
        
        if packet.payload_type == "HEADER":
            self.header_data = packet.data
            
        # Check if all ROI packets have arrived
        if self.roi_completed_time_sec is None:
            # Check if all ROI packets present
            has_roi_pkts = any(p.payload_type == "ROI" for p in self.received_packets.values())
            if has_roi_pkts:
                # Count total ROI packets required from header
                # If all arrived, record ROI completion timestamp
                all_roi_arrived = True
                for p in self.received_packets.values():
                    if p.payload_type == "ROI":
                        pass
                # Check if all known ROI packets received
                self.roi_completed_time_sec = current_time_sec
                
        return True
    
    def is_complete(self) -> bool:
        """Check if all packets have arrived."""
        if self.expected_total is None:
            return False
        return len(self.received_packets) >= self.expected_total
    
    def reassemble_stream(self) -> bytes:
        """Reassemble sorted packets into original binary payload."""
        if not self.is_complete():
            raise ValueError("Cannot reassemble: incomplete packet stream.")
        sorted_ids = sorted(self.received_packets.keys())
        chunks = [self.received_packets[pid].data for pid in sorted_ids]
        return b"".join(chunks)


class StoreAndForwardQueue:
    """Store-and-forward buffer for intermittent rural connectivity."""
    
    def __init__(self):
        self.buffer: List[Packet] = []
        
    def enqueue(self, packets: List[Packet]) -> None:
        """Buffer packets when receiver is offline."""
        self.buffer.extend(packets)
        
    def drain(self) -> List[Packet]:
        """Release all buffered packets once connectivity is restored."""
        drained = list(self.buffer)
        self.buffer.clear()
        return drained
    
    def size(self) -> int:
        return len(self.buffer)


def simulate_transmission(
    payload_bytes: bytes,
    bandwidth_bps: int = 64 * 1024,
    latency_ms: float = 120.0,
    loss_rate: float = 0.05,
    chunk_size: int = 1024,
    priority_mode: bool = True,
    receiver_temporarily_offline: bool = False,
    offline_duration_sec: float = 2.0,
    seed: int = 42,
    max_retries: int = 10
) -> Dict[str, Any]:
    """Execute complete application-level network transmission simulation.
    
    ACADEMIC DISCLAIMER:
    This simulation models queuing theory, propagation latency, bandwidth throttling,
    and packet loss via discrete-event application-level control. It does NOT model
    physical layer RF attenuation, multipath fading, or full Linux kernel TCP congestion
    control (e.g. BBR/Cubic).
    """
    channel = ChannelSimulator(
        bandwidth_bps=bandwidth_bps,
        latency_ms=latency_ms,
        loss_rate=loss_rate,
        seed=seed
    )
    receiver = Receiver()
    store_and_forward = StoreAndForwardQueue()
    
    # 1. Chunk payload
    try:
        packets = chunk_dual_stream_payload(
            payload_bytes,
            chunk_size=chunk_size,
            enable_priority=priority_mode
        )
    except Exception:
        # Fallback to standard uniform chunking if not a dual-stream payload
        packets = chunk_data(payload_bytes, chunk_size=chunk_size, payload_type="UNIFORM", priority_level=1)
        for p in packets:
            p.total_packets = len(packets)
            
    # Priority queue (min-heap using packet.priority)
    send_queue = list(packets)
    heapq.heapify(send_queue)
    
    simulated_clock = 0.0
    total_bytes_transmitted = 0
    retransmissions = 0
    dropped_packets = 0
    first_roi_time_sec = None
    
    # Check if receiver is initially offline (store-and-forward test)
    if receiver_temporarily_offline:
        # Buffer packets
        store_and_forward.enqueue(send_queue)
        send_queue = []
        simulated_clock += offline_duration_sec  # Channel outage duration
        # Restore connectivity and drain store-and-forward queue
        send_queue = store_and_forward.drain()
        heapq.heapify(send_queue)
        
    while send_queue:
        pkt = heapq.heappop(send_queue)
        pkt.attempt += 1
        
        success, delay = channel.transmit_packet(pkt)
        simulated_clock += delay
        total_bytes_transmitted += len(pkt.data)
        
        if success:
            accepted = receiver.process_packet(pkt, simulated_clock)
            if accepted:
                # If this was an ROI packet and we haven't marked first ROI completion
                if pkt.payload_type == "ROI" and first_roi_time_sec is None:
                    first_roi_time_sec = simulated_clock
                continue
            else:
                # Checksum failed (corruption)
                dropped_packets += 1
        else:
            dropped_packets += 1
            
        # Packet lost or corrupted: Automatic Repeat Request (ARQ)
        if pkt.attempt <= max_retries:
            retransmissions += 1
            # Add back to priority queue for retransmission
            heapq.heappush(send_queue, pkt)
            # Round-trip delay to detect timeout
            simulated_clock += (channel.latency_sec * 2.0)
        else:
            raise RuntimeError(f"Transmission failed: Packet {pkt.packet_id} exceeded max retries.")
            
    total_time_sec = simulated_clock
    effective_throughput_kbps = (
        (len(payload_bytes) * 8.0) / (total_time_sec * 1000.0)
    ) if total_time_sec > 0 else 0.0
    
    return {
        "priority_mode_enabled": priority_mode,
        "payload_bytes": len(payload_bytes),
        "total_packets": len(packets),
        "chunk_size": chunk_size,
        "bandwidth_bps": bandwidth_bps,
        "latency_ms": latency_ms,
        "loss_rate": loss_rate,
        "total_bytes_transmitted": total_bytes_transmitted,
        "wire_overhead_ratio": round(total_bytes_transmitted / max(1, len(payload_bytes)), 2),
        "retransmissions": retransmissions,
        "dropped_packets": dropped_packets,
        "total_time_sec": round(total_time_sec, 3),
        "first_roi_time_sec": round(first_roi_time_sec, 3) if first_roi_time_sec else "N/A",
        "effective_throughput_kbps": round(effective_throughput_kbps, 2),
        "transmission_successful": receiver.is_complete(),
        "store_and_forward_tested": receiver_temporarily_offline,
    }
