"""test_transmission.py: Unit tests for network simulation and priority transmission."""

import pytest

from src.transmission import (
    chunk_data,
    chunk_dual_stream_payload,
    compute_checksum,
    simulate_transmission,
    StoreAndForwardQueue,
    Receiver,
    Packet
)
from src.roi_compression import compress_roi_aware
from src.image_utils import generate_synthetic_cxr_phantom


def test_chunking_and_checksum():
    data = b"X" * 3500
    packets = chunk_data(data, chunk_size=1000, payload_type="TEST", priority_level=1)
    
    assert len(packets) == 4
    assert sum(len(p.data) for p in packets) == len(data)
    for p in packets:
        assert p.checksum == compute_checksum(p.data)


def test_receiver_reassembly():
    data = b"Telemedicine Diagnostic Data Stream Roundtrip"
    packets = chunk_data(data, chunk_size=10, payload_type="DATA", priority_level=1)
    for p in packets:
        p.total_packets = len(packets)
        
    receiver = Receiver()
    for p in packets:
        accepted = receiver.process_packet(p, current_time_sec=1.0)
        assert accepted is True
        
    assert receiver.is_complete()
    reassembled = receiver.reassemble_stream()
    assert reassembled == data


def test_receiver_corrupted_packet_rejection():
    data = b"Corrupted Transmission Sample"
    packets = chunk_data(data, chunk_size=10)
    receiver = Receiver()
    
    bad_pkt = packets[0]
    bad_pkt.data = b"AlteredPayload"  # Mismatch with checksum
    
    accepted = receiver.process_packet(bad_pkt, current_time_sec=1.0)
    assert accepted is False


def test_store_and_forward_queue():
    queue = StoreAndForwardQueue()
    data = b"Buffered store and forward payload"
    packets = chunk_data(data, chunk_size=10)
    
    assert queue.size() == 0
    queue.enqueue(packets)
    assert queue.size() == len(packets)
    
    drained = queue.drain()
    assert len(drained) == len(packets)
    assert queue.size() == 0


def test_transmission_simulation_with_loss_and_recovery():
    phantom, mask, _ = generate_synthetic_cxr_phantom(width=128, height=128, seed=42)
    _, payload, _ = compress_roi_aware(phantom, mask, q_roi=80, q_bg=20)
    
    # Run with 10% simulated loss and verify ARQ recovery
    results = simulate_transmission(
        payload_bytes=payload,
        bandwidth_bps=128 * 1024,
        latency_ms=50.0,
        loss_rate=0.10,
        chunk_size=1024,
        priority_mode=True,
        seed=100
    )
    
    assert results["transmission_successful"] is True
    assert results["retransmissions"] >= 0
    assert results["total_time_sec"] > 0
    assert results["effective_throughput_kbps"] > 0


def test_transmission_store_and_forward_simulation():
    phantom, mask, _ = generate_synthetic_cxr_phantom(width=128, height=128, seed=42)
    _, payload, _ = compress_roi_aware(phantom, mask, q_roi=80, q_bg=20)
    
    # Run with temporary receiver outage
    results = simulate_transmission(
        payload_bytes=payload,
        bandwidth_bps=128 * 1024,
        latency_ms=20.0,
        loss_rate=0.0,
        chunk_size=1024,
        priority_mode=True,
        receiver_temporarily_offline=True,
        offline_duration_sec=1.5,
        seed=42
    )
    
    assert results["transmission_successful"] is True
    assert results["total_time_sec"] >= 1.5
    assert results["store_and_forward_tested"] is True
