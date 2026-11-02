# Helix Relay — Product Specifications

Helix Relay is the flagship message-routing gateway from Helix Dynamics. This
page lists the specifications for the current generally available release.

## Current Version

The current generally available version is **Helix Relay 7.4**, released on
March 18, 2026. The previous long-term-support release is Helix Relay 6.9.

## Throughput and Performance

A single Helix Relay node sustains **48,000 messages per second** at a median
end-to-end latency of **9 milliseconds**. A cluster supports up to **64 nodes**,
for a maximum rated throughput of 3.0 million messages per second.

## Limits

- Maximum message payload size: **2 MB**
- Maximum number of routing rules per cluster: **5,000**
- Maximum concurrent client connections per node: **20,000**
- Default message retention: **72 hours**

## Deployment

Helix Relay ships as a single statically linked binary and as an official
container image. The minimum supported host has **4 vCPUs and 8 GB of RAM**.
Relay requires PostgreSQL 15 or later for its control-plane metadata store.

## Protocols

Helix Relay speaks three wire protocols natively: the Helix Wire Protocol (HWP),
MQTT 5.0, and AMQP 1.0. gRPC ingestion is available as an optional module.

## Licensing

Helix Relay is distributed under a commercial license. The license key is a
32-character token issued per cluster and must be renewed annually.
