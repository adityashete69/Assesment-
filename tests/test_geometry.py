from layout_detector.geometry import area, clip_box, iou, union


def test_clip_box_inside_image():
    assert clip_box((-10, -5, 2000, 900), 1000, 800) == (0, 0, 999, 799)


def test_clip_box_degenerate_is_made_valid():
    x0, y0, x1, y1 = clip_box((500, 500, 500, 500), 1000, 800)
    assert x1 > x0 and y1 > y0


def test_clip_box_fully_outside():
    x0, y0, x1, y1 = clip_box((1500, 900, 1600, 950), 1000, 800)
    assert 0 <= x0 < x1 <= 999 and 0 <= y0 < y1 <= 799


def test_iou_and_union():
    a, b = (0, 0, 10, 10), (5, 5, 15, 15)
    assert abs(iou(a, b) - 25 / 175) < 1e-9
    assert iou(a, (20, 20, 30, 30)) == 0.0
    assert union(a, b) == (0, 0, 15, 15)
    assert area(a) == 100
