#!/usr/bin/env python3
"""Quick check to verify the Sound Pad has the new features"""

print("\n" + "="*50)
print("SOUND PAD FEATURE VERIFICATION")
print("="*50 + "\n")

with open('oscilloscope_gui.py', 'r') as f:
    content = f.read()

features = [
    ("position_offset = tk.DoubleVar", "✓ Position Offset Variable"),
    ("Position Offset:", "✓ Position Offset Slider Control"),
    ("Time Slots (Click to select)", "✓ Time Slots Grid (16 buttons)"),
    ("Clear Selected Step", "✓ Clear Selected Step Button"),
    ("def clear_current_step", "✓ Clear Step Function"),
    ("Sound Pad Grid Settings", "✓ Grid Settings Scrollable Panel"),
]

all_found = True
for pattern, name in features:
    if pattern in content:
        print(name)
    else:
        print(f"✗ MISSING: {name}")
        all_found = False

print("\n" + "="*50)
if all_found:
    print("ALL FEATURES PRESENT!")
    print("\nTo see them:")
    print("1. python oscilloscope_gui.py")
    print("2. Click 'Sound Pad' button")
    print("3. Look at the RIGHT panel")
    print("4. Scroll down to see all controls")
else:
    print("SOME FEATURES MISSING - File may be wrong version")
print("="*50 + "\n")
