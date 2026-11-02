# Helix Dynamics — Information Security Policy

This policy governs authentication, access control, and data handling for all
Helix Dynamics systems. It is owned by the Security & Trust team.

## Passwords

All employee account passwords must be at least **14 characters** long and must
be rotated every **90 days**. The last **5 passwords** cannot be reused. Accounts
lock after **6 failed login attempts** and remain locked for 30 minutes.

## Multi-Factor Authentication (MFA)

MFA is **mandatory** for every employee on all internal systems. The only
approved MFA methods are hardware security keys (FIDO2) and the Helix
Authenticator mobile app. SMS-based one-time codes are **not permitted** as an
MFA factor.

## Device Encryption

All company laptops must use full-disk encryption. Laptops are automatically
locked after **10 minutes** of inactivity.

## Data Classification

Helix Dynamics uses four data classification tiers: **Public, Internal,
Confidential, and Restricted**. Restricted data (including customer encryption
keys) may only be accessed from managed devices on the corporate network.

## Incident Reporting

Suspected security incidents must be reported to the Security & Trust team within
**1 hour** of discovery via the `#security-incidents` channel or
security@helixdynamics.example. The security on-call pager is staffed 24/7.

## Access Reviews

Access to Restricted systems is reviewed **quarterly**. Any access unused for
**45 days** is automatically revoked.

## Vendor Software

New third-party software must pass a security review before installation.
Reviews are completed within **5 business days** of submission.
