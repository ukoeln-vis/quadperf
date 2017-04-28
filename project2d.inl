namespace QUAD_NS
{

namespace detail {

//-------------------------------------------------------------------------------------------------
// Misc. helpers
//

// remove max element from vec3 ---------------------------

template <
    typename T,
    typename = typename std::enable_if<simd::is_simd_vector<T>::value>::type
    >
VSNRAY_CPU_FUNC
vector<2, T> remove_max_element(vector<3, T> const& v)
{
    auto tmp = unpack(v);

    // Move max element of each vector to back
    for (size_t i = 0; i < tmp.size(); ++i)
    {
        auto& vv = tmp[i];
        auto max_idx = max_index(vv);

        if (max_idx == 0)
        {
            std::swap(vv.x, vv.z);
        }
        else if (max_idx == 1)
        {
            std::swap(vv.y, vv.z);
        }
    }

    return simd::pack(tmp).xy();
}

// remove element at index from vec3 ----------------------

template <
    typename T,
    typename = typename std::enable_if<simd::is_simd_vector<T>::value>::type,
    typename I // simd::int_type!
    >
VSNRAY_CPU_FUNC
vector<2, T> remove_at_index(vector<3, T> const& v, I const& index)
{
    using int_array = typename simd::aligned_array<I>::type;

    auto tmp = unpack(v);
    int_array idx;
    store(idx, index);

    for (size_t i = 0; i < tmp.size(); ++i)
    {
        auto& vv = tmp[i];

        if (idx[i] == 0)
        {
            std::swap(vv.x, vv.z);
        }
        else if (idx[i] == 1)
        {
            std::swap(vv.y, vv.z);
        }
    }

    return simd::pack(tmp).xy();
}

VSNRAY_FUNC
vec2 remove_at_index(vec3 const& v, int const& index)
{
    if (index == 0)
        return vec2(v.y, v.z);

    if (index == 1)
        return vec2(v.x, v.z);

    if (index == 2)
        return vec2(v.x, v.y);
}


// SIMD max_index for vec3 --------------------------------

template <
    typename T,
    typename = typename std::enable_if<simd::is_simd_vector<T>::value>::type
    >
VSNRAY_CPU_FUNC
inline auto max_index(vector<3, T> const& u)
    -> typename simd::int_type<T>::type
{
    using I = typename simd::int_type<T>::type;
    using int_array = typename simd::aligned_array<I>::type;

    int_array i;

    auto tmp = unpack(u);

    for (size_t d = 0; d < tmp.size(); ++d)
    {
        i[d] = select(tmp[d].y < tmp[d].x, 0, 1);
        i[d] = select(tmp[d][i[d]] < tmp[d].z, 2, i[d]);
    }

    return I(i);
}


template <typename T>
struct line
{
    VSNRAY_FUNC line() = default;
    VSNRAY_FUNC line(vector<2, T> const& vv1, vector<2, T> const& vv2)
        : v1(vv1)
        , v2(vv2)
    {
    }

    vector<2, T> v1;
    vector<2, T> v2;
};

// intersect two line segments (1st is parallel to y-axis)!
template <typename T>
VSNRAY_CPU_FUNC
inline auto intersect(vector<2, T> const& p, line<T> const& l)
    -> typename simd::int_type<T>::type
{
    using I = typename simd::int_type<T>::type;

    auto a = l.v1;
    auto b = l.v2;

    return select(
            ((a.x < p.x && p.x < b.x) && ((b.y * (a.x - p.x) + a.y * (p.x - b.x)) >= (a.x - b.x) * p.y))
         || ((b.x < p.x && p.x < a.x) && ((b.y * (a.x - p.x) + a.y * (p.x - b.x)) <= (a.x - b.x) * p.y)),
            I(1),
            I(0)
            );
}


//-------------------------------------------------------------------------------------------------
// intersect by projecting quad edges to 2D (cf. Glassner 1989,
//      "An Introduction to Ray Tracing", p. 55
//

template <typename R>
VSNRAY_FUNC
hit_record<R, primitive<unsigned>> intersect_project_2D(R const& ray, basic_quad<float> const& quad)
{
    using T  = typename R::scalar_type;
    using I  = typename simd::int_type<T>::type;
    using V  = vector<3, T>;
    using V2 = vector<2, T>;
    using L  = line<T>;

    hit_record<R, primitive<unsigned>> result;

    V v1(quad.v1);
    V v2(quad.v2);
    V v3(quad.v3);
    V v4(quad.v4);

    V e1(v2 - v1);
    V e2(v3 - v2);

    V n = normalize(cross(e1, e2));

    T div = dot(n, ray.dir);

    // ray/plane intersection
    T t = select(
            div != T(0.0),
            dot(v1 - ray.ori, n) / div,
            t
            );

    result.hit = div != T(0.0) && t >= T(0.0);
    result.t = t;

    if (!any(result.hit))
    {
        return result;
    }

    V isect_pos = ray.ori + ray.dir * t;


    // Project to 2D by throwing away max component of plane eq.
    I index = max_index(n);

    V2 ip_2 = remove_at_index(isect_pos, index);

    V2 v1_2 = remove_at_index(v1, index);
    V2 v2_2 = remove_at_index(v2, index);
    V2 v3_2 = remove_at_index(v3, index);
    V2 v4_2 = remove_at_index(v4, index);

    I num_intersections(0);
    num_intersections += intersect(ip_2, L(v1_2, v2_2));
    num_intersections += intersect(ip_2, L(v2_2, v3_2));
    num_intersections += intersect(ip_2, L(v3_2, v4_2));
    num_intersections += intersect(ip_2, L(v4_2, v1_2));

    result.hit &= num_intersections == 1;
    // result.prim_id = quad.prim_id;
    // result.geom_id = quad.geom_id;
    result.isect_pos = isect_pos;

#if CALCULATE_UV
    if (!any(result.hit))
    {
        return result;
    }

    V2 uv = get_uv(quad, isect_pos);
    result.u = uv.x;
    result.v = uv.y;
#endif

    return result;
}


} // detail

} // QUAD_NS
