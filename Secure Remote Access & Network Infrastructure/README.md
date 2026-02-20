# Secure Remote Access & Network Infrastructure (Home Lab)

## Project Objective
The goal of this home lab project was to design and implement a secure, encrypted network infrastructure that allows remote access to internal resources without exposing sensitive services (like RDP) to the public internet. 

By leveraging **OpenVPN** and **DHCP Static IP Reservations** on a Gigabit router, I established a Zero Trust approach to remote management, completely eliminating public port exposure.

---

##  Technologies & Hardware Used
* **Hardware:** Asus Gigabit Router (RT-BE50), Local Workstation/Server.
* **Protocols & Services:** OpenVPN, UDP/TCP, DHCP, RDP (Remote Desktop Protocol).
* **Security Concepts:** Tunneling, Encryption, Zero Trust, Port Hiding, MAC Binding.

---

## Implementation Steps & Architecture

### 1. OpenVPN Server Configuration
I configured the Asus router to act as an OpenVPN server. This creates a secure, encrypted tunnel between my remote device (client) and the home network, routing all traffic through the VPN.
* **Protocol:** UDP (for faster performance) / Port 1194.
* **Encryption:** AES-based encryption configured via the router's VPN interface.


### 2. DHCP Static IP Reservation (MAC Binding)
To ensure the target workstation always receives the same local IP address, I configured a DHCP binding rule. The router matches the workstation's MAC address and permanently assigns it a specific local IP (e.g., `192.168.x.x`).
* This is crucial for reliable remote access, as the RDP connection relies on a fixed, predictable IP.

### 3. Securing Remote Desktop (RDP) via Zero Trust
Instead of opening port 3389 on the router's WAN (which is highly vulnerable to brute-force attacks), RDP is completely blocked from the outside. 
* To access the workstation, the remote client must first establish the VPN connection.
* Once inside the VPN tunnel, the client uses the workstation's local, static IP to initiate the RDP session securely.

---

##  Key Takeaways & Cloud Relevance
While implemented on physical home hardware, this project heavily mirrors fundamental **Cloud Engineering** concepts:
* **VPC & Subnetting:** Managing private IP spaces and DHCP allocation.
* **Bastion Hosts / Client VPN:** Using a secure entry point (the router/VPN) to access internal network instances (the workstation) without public IP exposure.
* **Security Groups / Firewalls:** Denying all inbound traffic by default and allowing access only via secure, authenticated tunnels.