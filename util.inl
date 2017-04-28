using namespace visionaray;

template <typename T, typename U, typename S>
VSNRAY_FUNC
inline U slerp(
        T p1,
        T p2,
        U v1,
        U v2,
        S t
        )
{
    // angle between p1, p2
    auto w = acos(dot(p1, p2));

    // edge case
    auto average = (v1 + v2) * U(0.5);

    auto interpolated = (sin((S(1.) - t) * w) * v1 + sin(t * w) * v2) / sin(w);

    return select(w == S(0.), average, interpolated);
}
