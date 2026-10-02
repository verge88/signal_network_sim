"""Shared node symbols for the topology canvas and vector export."""
import math
from html import escape


def node_shape(protocol, node_type):
    kind = node_type.upper()
    # SCP is a service control point in SS7, but a proxy in 5G SBA.
    if kind == "SCP":
        return "router" if protocol == "5g_sba" else "database"
    if kind in {"HLR", "VLR", "HSS", "UDM", "UDR", "NRF", "REGISTRAR"}:
        return "database"
    if kind in {"STP", "DRA", "PROXY", "REDIRECT"}:
        return "router"
    if kind in {"IGW", "DEA", "SGW", "PGW", "SBC", "UA_GW"}:
        return "gateway"
    if kind in {"SP", "MME", "PCRF", "AMF", "SMF", "AUSF", "PCF"}:
        return "server"
    return "endpoint"


def symbol_parts(protocol, node_type, x, y, r):
    """Return filled outlines and decorative lines within the node bounds."""
    shape = node_shape(protocol, node_type)
    def coords(points):
        return tuple(v for px, py in points for v in (x + px*r, y + py*r))
    if shape == "database":
        bottom = [(math.cos(a), .72 + .28*math.sin(a))
                  for a in [i*math.pi/24 for i in range(25)]]
        return [("polygon", coords([(-1, -.72), (1, -.72)] + bottom)),
                ("oval", (x-r, y-r, x+r, y-.44*r))]
    if shape == "router":
        return [("polygon", coords([(-1, 0), (-.55, -1), (.55, -1),
                                     (1, 0), (.55, 1), (-.55, 1)]))]
    if shape == "gateway":
        return [("polygon", coords([(-.65, -1), (.65, -1), (1, -.65),
                                     (1, .65), (.65, 1), (-.65, 1),
                                     (-1, .65), (-1, -.65)])),
                ("line", coords([(-.62, -.6), (-.62, .6)])),
                ("line", coords([(.62, -.6), (.62, .6)]))]
    if shape == "server":
        return [("rectangle", (x-.8*r, y-r, x+.8*r, y+r)),
                ("line", coords([(-.55, .52), (.55, .52)])),
                ("line", coords([(-.55, .75), (.15, .75)]))]
    return [("oval", (x-r, y-r, x+r, y+r))]


def draw_symbol(canvas, protocol, node_type, x, y, r, fill, outline, width):
    for kind, points in symbol_parts(protocol, node_type, x, y, r):
        options = dict(fill=outline, width=width) if kind == "line" else dict(
            fill=fill, outline=outline, width=width)
        getattr(canvas, "create_" + kind)(*points, **options)


def symbol_svg(protocol, node_type, x, y, r, fill, outline, width):
    result = []
    style = (f'fill="{escape(fill, quote=True)}" '
             f'stroke="{escape(outline, quote=True)}" stroke-width="{width:.2f}" '
             'stroke-linejoin="round"')
    for kind, p in symbol_parts(protocol, node_type, x, y, r):
        if kind == "oval":
            result.append(f'<ellipse cx="{(p[0]+p[2])/2:.2f}" cy="{(p[1]+p[3])/2:.2f}" '
                          f'rx="{(p[2]-p[0])/2:.2f}" ry="{(p[3]-p[1])/2:.2f}" {style}/>')
        elif kind == "rectangle":
            result.append(f'<rect x="{p[0]:.2f}" y="{p[1]:.2f}" '
                          f'width="{p[2]-p[0]:.2f}" height="{p[3]-p[1]:.2f}" {style}/>')
        else:
            points = " ".join(f"{p[i]:.2f},{p[i+1]:.2f}" for i in range(0, len(p), 2))
            tag = "polyline" if kind == "line" else "polygon"
            result.append(f'<{tag} points="{points}" {style}/>')
    return result


def symbol_contains(protocol, node_type, dx, dy, r):
    if r <= 0:
        return False
    shape = node_shape(protocol, node_type)
    u, v = abs(dx/r), abs(dy/r)
    if shape == "endpoint":
        return u*u + v*v <= 1
    if shape == "server":
        return u <= .8 and v <= 1
    if shape == "router":
        return v <= 1 and u <= 1 - .45*v
    if shape == "gateway":
        return u <= 1 and v <= 1 and u+v <= 1.65
    return u <= 1 and (v <= .72 or u*u + ((v-.72)/.28)**2 <= 1)
