# Security policy

## Supported versions

Security fixes are applied to the latest commit on `main`. There are no releases,
no tags and no binaries: the project is in design phase and nothing here runs yet.
Forks and modified copies are not supported.

TinyTitan Datacenter is a research project. It is not intended for production,
multi-user, or security-critical deployments.

## Scope, stated honestly

Because no runtime exists yet, the security surface today is **the repository
itself**: documents, tooling, CI configuration and the wiki. Once a cluster runtime
exists, its posture is part of the design and is deliberately narrow:

- the cluster exchanges data between **its own nodes** over a trusted, local network;
- it has no authentication or transport encryption of its own, and it is not
  designed to be reachable from another network;
- nothing — no model, no API, no control channel — is intended to be exposed beyond
  the farm's own machines.

Anything that contradicts that posture is a security finding, not a feature request.

## Reporting a vulnerability

Do not open a public issue for a suspected security vulnerability. Use
[GitHub private vulnerability reporting](https://github.com/Pummelchen/TinyTitan_Datacenter/security/advisories/new)
instead.

Include:

- a description of the vulnerability;
- the affected commit or version;
- reproduction steps or a minimal proof of concept;
- the expected and observed behaviour;
- the potential security impact; and
- a suggested mitigation, if you know one.

**Do not include personal data, credentials, access tokens, host addresses, or
copyrighted model weights in the report.** A leaked credential is itself a finding:
report it privately and immediately, and never paste the value into an issue, a
commit, a pull request or the wiki.

Reports are especially useful when they involve credential exposure, unsafe
model-package handling, path traversal, buffer or offset safety, malformed remote
data, verification or attestation bypasses, command-line injection, or unexpected
file access. Model quality, incorrect generated text, expected high resource use and
performance regressions are not security vulnerabilities.

Please allow the issue to be investigated and a fix prepared before publishing
details. Credit can be included in the eventual advisory unless you prefer to remain
anonymous.
