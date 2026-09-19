<img width="1254" height="1254" alt="TinyTitanDatacenter" src="TinyTitanDatacenter.png" />



# TinyTitan Datacenter

[![Stars](https://img.shields.io/github/stars/Pummelchen/TinyTitan_Datacenter?style=flat-square&logo=github&label=Stars&color=e3b341)](https://github.com/Pummelchen/TinyTitan_Datacenter/stargazers)
[![Views (14d)](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/Pummelchen/TinyTitan_Datacenter/main/.github/traffic.json)](https://github.com/Pummelchen/TinyTitan_Datacenter)
[![Last Commit](https://img.shields.io/github/last-commit/Pummelchen/TinyTitan_Datacenter?style=flat-square&logo=git&label=Last%20Commit&color=2ea44f)](https://github.com/Pummelchen/TinyTitan_Datacenter/commits/main)
[![Contact](https://img.shields.io/badge/Contact-0xa0b1%40gmail.com-blue?style=flat-square&logo=gmail&logoColor=white)](mailto:0xa0b1@gmail.com)


## Project Target

Run large >120B MOE LLM on a distributed network of Mac Mini/Studio's using **expert parallelism** with SSD streaming to reduce RAM requirements.

## Project Status

- Single nodes in a 4x Mac Mini cluster exceed decode tok/s over the sister project TinyTitan by 10-15% so the new engine build from scratch is performing better than expected.
- The network stack is working and performing well over raw TCP on 1GBit LAN.
- Cluster tok/s is still below a single node - the core work in this project.

## License

MIT — see [LICENSE](LICENSE). Copyright (c) 2026 André Borchert.

## Contact

Questions, bug reports and suggestions are always welcome. You can contact André Borchert by email at [0xa0b1@gmail.com](mailto:0xa0b1@gmail.com).
