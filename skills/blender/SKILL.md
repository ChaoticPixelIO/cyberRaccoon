---
name: blender
description: Blender 3D creation suite — window layout, keyboard shortcuts, and common modeling/rendering workflows.
---

# Blender Skill

You are controlling Blender, a 3D creation suite. Use this knowledge to navigate the UI and perform actions efficiently.

## Window Layout (Default)

- **Top**: Menu bar (File, Edit, Render, Window, Help) + workspace tabs (Layout, Modeling, Sculpting, UV Editing, Texture Paint, Shading, Animation, Rendering, Compositing, Geometry Nodes)
- **Center-left**: 3D Viewport (main working area)
- **Right**: Properties panel (vertical tabs: Active Tool, Render, Output, View Layer, Scene, World, Object, Modifiers, Particles, Physics, Constraints, Object Data)
- **Top-right of 3D Viewport**: Outliner (scene hierarchy tree)
- **Bottom**: Timeline (animation keyframes)
- **Left edge of 3D Viewport**: Toolbar (T to toggle)

## Essential Keyboard Shortcuts

### Navigation (in 3D Viewport)
- **Middle mouse drag**: Orbit view
- **Shift+Middle mouse drag**: Pan view
- **Scroll wheel**: Zoom in/out
- **Numpad 1/3/7**: Front/Right/Top orthographic view
- **Numpad 5**: Toggle perspective/orthographic
- **Numpad 0**: Camera view
- **Home**: Frame all objects
- **Numpad .**: Frame selected object

### Selection
- **Left click**: Select object
- **Shift+Left click**: Add to selection
- **A**: Select all / Deselect all (toggle)
- **B**: Box select (drag rectangle)
- **C**: Circle select (paint selection)
- **Alt+A**: Deselect all

### Transform
- **G**: Grab/Move — then X/Y/Z to constrain to axis, type number for precise distance
- **R**: Rotate — then X/Y/Z to constrain, type degrees
- **S**: Scale — then X/Y/Z to constrain, type factor
- **Shift+D**: Duplicate
- **Right click** or **Esc**: Cancel current operation
- **Left click** or **Enter**: Confirm current operation

### Mode Switching
- **Tab**: Toggle Edit Mode / Object Mode
- **1/2/3** (top row, in Edit Mode): Vertex/Edge/Face select mode
- **Ctrl+Tab**: Mode pie menu (Object, Edit, Sculpt, etc.)

### Editing
- **X** or **Delete**: Delete (shows confirmation menu)
- **E**: Extrude (in Edit Mode)
- **I**: Inset faces (in Edit Mode)
- **Ctrl+R**: Loop cut (in Edit Mode)
- **K**: Knife tool (in Edit Mode)
- **F**: Make face/edge (in Edit Mode)
- **M**: Merge vertices (in Edit Mode)
- **P**: Separate selection (in Edit Mode)
- **Ctrl+J**: Join selected objects (in Object Mode)

### Common Operations
- **Shift+A**: Add menu (Mesh, Curve, Surface, etc.)
- **Ctrl+Z**: Undo
- **Ctrl+Shift+Z**: Redo
- **Ctrl+S**: Save
- **F2**: Rename selected object
- **N**: Toggle sidebar (properties panel in viewport)
- **T**: Toggle toolbar
- **F3**: Search commands (type any operation name)
- **Z**: Shading pie menu (Wireframe, Solid, Material Preview, Rendered)

### Rendering
- **F12**: Render image
- **Ctrl+F12**: Render animation
- **Esc**: Cancel render

## Workflow Tips

- **To add a primitive**: Press Shift+A, navigate to Mesh submenu, select shape (Cube, Sphere, Cylinder, etc.)
- **To apply transforms**: Ctrl+A → select what to apply (Location, Rotation, Scale, All)
- **To set origin**: Right-click object → Set Origin → choose option
- **To enter precise values**: After starting a transform (G/R/S), type the number directly. E.g., G Z 5 Enter moves up 5 units.
- **To snap to grid**: Hold Ctrl during transform
- **To smooth shade**: Right-click object → Shade Smooth
- **Search for any command**: Press F3 and type the command name — this is the fastest way to find obscure features
- **Properties panel shortcuts**: Click the vertical tab icons on the right to switch between Render, Object, Modifier, Material, etc. properties

## Common Pitfalls

- Objects may appear invisible if they're on a hidden collection — check the Outliner (eye icon)
- If transforms seem wrong, check that you've applied scale (Ctrl+A → Scale) before adding modifiers
- "Can't edit" errors usually mean you're in the wrong mode — press Tab to toggle Edit Mode
- If the viewport seems frozen, check if you're in camera view (Numpad 0) and press Numpad 0 again to exit
- Blender uses right-click context menus extensively — right-click objects and UI elements for options
