# BIOS/UEFI Operation Skill

You are controlling a computer at the BIOS/UEFI firmware level. This is a pre-OS environment with completely different rules from normal desktop operation.

## V1 Constraints — Read Before Proceeding

- **UEFI only.** Legacy BIOS is NOT supported. You MUST verify UEFI before rebooting.
- **Windows and Linux only.** macOS is NOT supported.
- **Running OS required.** The target machine must have a working OS to reboot from.

## Step 1: Open a Terminal

Open a terminal window that stays open so you can read command output from the screen.

**Windows:** Open PowerShell by pressing `Win+X` then selecting "Windows PowerShell" or "Terminal". If you need admin privileges (for the reboot command later), right-click the Start button and select "Windows PowerShell (Admin)" or "Terminal (Admin)". A Windows UAC prompt will appear — click "Yes" (是) to approve it. This is expected and does not require user intervention.

**Linux:** Open a terminal with `Ctrl+Alt+T` or by finding Terminal in the application menu.

IMPORTANT: Do NOT use `powershell -Command "..."` from the Run dialog. The window closes immediately and you cannot read the output. Always open a terminal window first, then type commands inside it.

## Step 2: Verify UEFI

In the terminal you just opened, type the UEFI detection command and read the output.

**Linux:**
```
[ -d /sys/firmware/efi ] && echo "FIRMWARE: UEFI" || echo "FIRMWARE: LEGACY_BIOS"
```

**Windows (type this inside the admin PowerShell window):**
```
bcdedit | Select-String "path"
```

Read the output from the screen. If the path values end with `.efi` (e.g., `\EFI\Microsoft\Boot\bootmgfw.efi`), the machine is UEFI. If they end with `.exe` (e.g., `\Windows\system32\winload.exe`), it is Legacy BIOS.

If the output shows Legacy BIOS (`.exe` paths), report: "This machine uses Legacy BIOS, which is not currently supported for automated BIOS operation." Then stop.

## Step 3: Detect Motherboard Vendor (Optional)

In the same terminal, type the vendor detection command and read the output.

**Linux:**
```
cat /sys/class/dmi/id/board_vendor && cat /sys/class/dmi/id/board_name
```

**Windows (type this inside the PowerShell window):**
```
wmic baseboard get manufacturer,product
```

## Step 4: Reboot Into UEFI Firmware

In the same terminal window, type the reboot command:

**Windows (requires admin terminal):**
```
shutdown /r /fw /t 0
```

**Linux:**
```
sudo systemctl reboot --firmware-setup
```

After typing the reboot command, the screen will go black for several seconds. This is normal. Wait for the BIOS/UEFI setup screen to appear.

## BIOS Navigation Rules

Once you are in the BIOS/UEFI setup screen:

1. **Keyboard ONLY.** Most BIOS menus do not support mouse input. Use arrow keys, Enter, and Escape exclusively.
2. **No clipboard.** You cannot paste text. Type everything character by character.
3. **No terminal.** There is no command line, no taskbar, no window manager.
4. **Read the screen carefully.** BIOS screens show navigation hints at the bottom or side (e.g., "F10: Save & Exit", "Esc: Back", "+/-: Change Value").

### Common BIOS Keys

| Key | Action |
|-----|--------|
| Up/Down arrows | Move between menu items |
| Left/Right arrows | Switch between top-level tabs |
| Enter | Select/enter a submenu or toggle a setting |
| Esc | Go back / cancel |
| F10 | Save changes and exit (most vendors) |
| +/- | Change selected value (some vendors) |
| F9 | Load default settings (some vendors) |

### Typical BIOS Menu Structure

Most UEFI BIOS setups have a tab-based layout:
- **Main** — System info, date/time
- **Advanced** — CPU, storage, USB, network settings
- **Security** — Secure Boot, passwords, TPM
- **Boot** — Boot order, boot mode (UEFI/Legacy)
- **Save & Exit** — Save changes, discard changes, load defaults

Navigate between tabs with Left/Right arrow keys. Navigate within a tab with Up/Down arrow keys.

### How to Change a Setting

1. Navigate to the correct tab (Left/Right arrows)
2. Navigate to the setting (Up/Down arrows)
3. Press Enter to select it
4. Choose the new value (Up/Down or +/-, then Enter to confirm)
5. After making all changes: press F10 to Save & Exit
6. Confirm "Yes" when prompted

## Step 5: Save and Return to OS

After changing the desired setting:
1. Press **F10** (Save & Exit)
2. Select **Yes** to confirm saving
3. The machine will reboot back into the operating system
4. Wait for the OS desktop or login screen to appear
5. Confirm the task is complete

## Vendor-Specific BIOS Entry Keys (Reference)

This table is for future use when POST key interception is supported.

| Vendor | BIOS Key | Boot Menu Key | Notes |
|--------|----------|---------------|-------|
| ASUS (desktop) | Del | F8 | UEFI graphical interface, may support mouse |
| ASUS (laptop) | F2 | Esc | |
| MSI | Del | F11 | Click BIOS — graphical, mouse support |
| Gigabyte | Del | F12 | |
| ASRock | Del or F2 | F11 | |
| Dell | F2 | F12 | |
| HP | F10 | F9 or Esc | |
| Lenovo ThinkPad | F1 | F12 | May require Fn+F1 |
| Lenovo IdeaPad | F2 | F12 | |
| Acer | F2 | F12 | |
| Intel NUC | F2 | F10 | |
| Samsung | F2 | F12 or Esc | |
| Toshiba/Dynabook | F2 | F12 | |
| Sony VAIO | F2 | F11 | |
| Fujitsu | F2 | F12 | |
| Microsoft Surface | Volume Up (held during power on) | — | Special case |

## Safety Rules

**NEVER change these settings unless the user explicitly requests them:**
- Memory frequency, timing, or voltage (XMP/DOCP profiles)
- CPU voltage, multiplier, or overclocking settings
- Fan curve or thermal settings
- Boot mode (UEFI to Legacy or vice versa) — can make the OS unbootable
- Secure Boot keys (adding/removing custom keys)
- TPM clear or firmware TPM toggle — can lock out BitLocker volumes
- BIOS password — can lock out the user permanently

**ALWAYS confirm before saving (F10).** Double-check that the setting shown on screen matches what was requested.

If you are unsure about a setting or its consequences, report: "I found the setting but I'm not confident about the correct value. Please verify: [describe what you see]." Then stop.
