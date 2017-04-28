#pragma once

#include <visionaray/math/math.h>

#include <visionaray/tags.h>
#include <visionaray/get_normal.h>
#include <visionaray/get_shading_normal.h>
#include <visionaray/get_tex_coord.h>
#include <visionaray/bvh.h> // FIXME needs to be included before intersector.h
#include <visionaray/intersector.h>

using namespace visionaray;

namespace QUAD_NS
{

template <typename T>
class quad_prim //: public visionaray::primitive<unsigned>
{
public:

    using scalar_type =  T;
    using vec_type    =  vector<3, T>;

public:

    // vertex ordering:
    //
    //    v3
    //       x
    //       | \
    //       |   \
    //       |     \
    //       |       \
    //       |         \
    //       |           x  v4
    //       |           |
    //       |           |
    //       |           |
    //       |           |
    //       x-----------x
    //    v1                v2
    //
    MATH_FUNC quad_prim() = default;
    MATH_FUNC quad_prim(
            vector<3, T> const& v1,
            vector<3, T> const& v2,
            vector<3, T> const& v3,
            vector<3, T> const& v4
            )
        : v1(v1)
        , e1(v2-v1)
        , e2(v3-v1)
    {
        // if this quad should be degenerated to a triangle, make sure that v3 == v4
        // v2 == v4 would also be possible, but would need additional checks
        // and a rewrite of the following code

        // edge case, degenerated quad
        if (v3 == v4)
        {
            // ensure this quad is exactly a triangle
            this->v4= vector<2, T>(0., 1.);
            return;
        }

        // calculate (u,v)-coordinates of the 4th vector relative to the triangle (v1, e1, e2)

        vec_type d = v4-v1;
        vec_type s2 = cross(d, e1); // this is a normal of the triangle, use as ray dir

        vec_type s1 = cross(s2, e2);
        scalar_type div = dot(s1, e1);
        scalar_type inv_div = T(1.0) / div;

        scalar_type b1 = dot(d, s1) * inv_div;
        scalar_type b2 = dot(s2, s2) * inv_div;

        this->v4 = vector<2, T>(b1, b2);
    }

    vec_type v1;
    vec_type e1;
    vec_type e2;
    vector<2, scalar_type> v4;

    // need to reorder vertices for degenerated quads
    //
    // vertex ordering:
    //
    //    v4
    //       x
    //       | \
    //       |   \
    //       |     \
    //       |       \
    //       |         \
    //       |           x  v3
    //       |           |
    //       |           |
    //       |           |
    //       |           |
    //       x-----------x
    //    v1                v2
    //
    static quad_prim<float> make_quad(
                vector<3, float> const& v1,
                vector<3, float> const& v2,
                vector<3, float> const& v3,
                vector<3, float> const& v4,
                float epsilon=0.01f
            )
    {
        // ensure degenerated quads are exactly a triangles and the matching
        // vertices are v3 and v4
        if (norm(v1-v2) < epsilon)
            return quad_prim<float>(v3, v4, v2, v2);
        else if (norm(v2-v3) < epsilon)
            return quad_prim<float>(v4, v1, v3, v3);
        else if (norm(v3-v4) < epsilon)
            return quad_prim<float>(v1, v2, v4, v4);
        else if (norm(v4-v1) < epsilon)
            return quad_prim<float>(v2, v3, v1, v1);

        else
            return quad_prim<float>(v1, v2, v4, v3);
    }
};

namespace detail
{

template <typename T, typename U>
MATH_FUNC
inline hit_record<basic_ray<T>, primitive<unsigned>> intersect_opt(
        basic_ray<T> const&                     ray,
        QUAD_NS::quad_prim<U> const&               quad
        )
{

    typedef vector<3, T> vec_type;

    hit_record<basic_ray<T>, primitive<unsigned>> result;
    result.t = T(-1.0);

    // case T != U
    vec_type v1(quad.v1);
    vec_type e1(quad.e1);
    vec_type e2(quad.e2);
    vector<2, T> v4(quad.v4);

    vec_type s1 = cross(ray.dir, e2);
    T div = dot(s1, e1);

    result.hit = ( div != T(0.0) );

    if ( !any(result.hit) )
    {
        return result;
    }

    T inv_div = T(1.0) / div;

    vec_type d = ray.ori - v1;
    T b1 = dot(d, s1) * inv_div;

    result.hit &= ( b1 >= T(0.0) );

    if ( !any(result.hit) )
    {
        return result;
    }

    vec_type s2 = cross(d, e1);
    T b2 = dot(ray.dir, s2) * inv_div;

    result.hit &= ( b2 >= T(0.0) );

    if ( !any(result.hit) )
    {
        return result;
    }

    // (b1,b2) are (u,v)-coordinates (relative to the triangle (v1,e1,e2)) of
    // the intersection point - check if they lie inside quad

    result.hit &= ( b2 <= ((v4.y-T(1.0)) / v4.x) * b1 + T(1.0) ) || v4.x == T(0.0);
    result.hit &= ( b1 <= ((v4.x-T(1.0)) / v4.y) * b2 + T(1.0) );

    if ( !any(result.hit) )
    {
        return result;
    }

#if CALCULATE_UV
    // now calculate bilinear coordinates relative to the quad
    T u, v;

    // special cases:
    if (quad.v4.x == T(1.0))
    {
        u = b1;
        v = b2 / (u * (v4.y - T(1.0)) + T(1.0));
    }
    else if (quad.v4.y == T(1.0))
    {
        v = b2;
        u = b1 / (v * (v4.x - T(1.0)) + T(1.0));
    }
    else
    {
        // solve A*u^2 + B*u + C = 0
        T A = -(v4.y - T(1.0));
        T B = b1 * (v4.y - T(1.0)) - b2 * (v4.x - T(1.0)) - T(1.0);
        T C = b1;

        T D = B * B - T(4.0) * A * C;
        T Q = -T(0.5) * (B + copysign(sqrt(D), B));

        u = Q / A;
        u = select(u < T(0.0) || u > T(1.0), C / Q, u);
        v = b2 / (u * (v4.y - T(1.0)) + T(1.0));
    }

    result.u = u;
    result.v = v;
#endif

    // result.prim_id = quad.prim_id;
    // result.geom_id = quad.geom_id;
    result.t = dot(e2, s2) * inv_div;
    return result;

}

} // detail

template <typename T, typename U>
MATH_FUNC
inline hit_record<basic_ray<T>, primitive<unsigned>> intersect(
        basic_ray<T> const&                     ray,
        QUAD_NS::quad_prim<U> const&               quad
        )
{
    return detail::intersect_opt(ray, quad);
}

struct quad_intersector_opt : basic_intersector<quad_intersector_opt>
{
    using basic_intersector<quad_intersector_opt>::operator();

    template <typename R, typename S>
    VSNRAY_FUNC
    auto operator()(
            R const& ray,
            quad_prim<S> const& quad
            )
        -> decltype( detail::intersect_opt(ray, quad) )
    {
        return detail::intersect_opt(ray, quad);
    }
};

} // QUAD_NS


namespace visionaray
{


template <typename T>
MATH_FUNC
basic_aabb<T> get_bounds(QUAD_NS::quad_prim<T> const& t)
{
    basic_aabb<T> bounds;

    auto v1 = t.v1;
    auto v2 = v1 + t.e1;
    auto v3 = v1 + t.e2;
    auto v4 = (1 - t.v4.x - t.v4.y) * v1 + t.v4.x * v2 + t.v4.y * v3;

    bounds.invalidate();
    bounds.insert(v1);
    bounds.insert(v2);
    bounds.insert(v3);
    bounds.insert(v4);

    return bounds;
}

template <typename T>
void split_primitive(aabb& L, aabb& R, float plane, int axis, QUAD_NS::quad_prim<T> const& prim)
{
    auto v1 = prim.v1;
    auto v2 = v1 + prim.e1;
    auto v3 = v1 + prim.e2;
    auto v4 = (1 - prim.v4.x - prim.v4.y) * v1 + prim.v4.x * v2 + prim.v4.y * v3;

    L.invalidate();
    R.invalidate();

    detail::split_edge(L, R, v1, v2, plane, axis);
    detail::split_edge(L, R, v2, v3, plane, axis);
    detail::split_edge(L, R, v3, v4, plane, axis);
    detail::split_edge(L, R, v4, v1, plane, axis);
}

template <typename HR, typename T>
VSNRAY_FUNC
inline vector<3, T> get_normal(HR const& hr, QUAD_NS::quad_prim<T> const& quad)
{
    VSNRAY_UNUSED(hr);

    return normalize(cross(quad.e1, quad.e2));
}

template <typename Normals, typename HR, typename T>
VSNRAY_FUNC
inline auto get_normal(
        Normals                     normals,
        HR const&                   hr,
        QUAD_NS::quad_prim<T>       prim,
        per_vertex_binding          /* */
        )
    -> decltype( get_normal(hr, prim) )
{
    VSNRAY_UNUSED(normals);

    return get_normal(hr, prim);
}

template <typename Normals, typename HR, typename T>
VSNRAY_FUNC
inline auto get_shading_normal(
        Normals                     normals,
        HR const&                   hr,
        QUAD_NS::quad_prim<T>       /* */,
        per_vertex_binding          /* */
        )
    -> typename std::iterator_traits<Normals>::value_type
{
    auto v1 = normals[hr.prim_id * 4];
    auto v2 = normals[hr.prim_id * 4 + 1];
    auto v3 = normals[hr.prim_id * 4 + 2];
    auto v4 = normals[hr.prim_id * 4 + 3];

//    return (1-hr.u) * (1-hr.v) * v1
//         + hr.u     * (1-hr.v) * v2
//         + hr.u     * hr.v     * v3
//         + (1-hr.u) * hr.v     * v4;

    auto a = slerp(v1, v2, v1, v2, hr.u);
    auto b = slerp(v4, v3, v4, v3, hr.u);

    auto r = slerp(a, b, a, b, hr.v);

    return r;
}


template <typename T>
struct num_vertices<QUAD_NS::quad_prim<T>>
{
    enum { value = 4 };
};
template <typename T>
struct num_normals<QUAD_NS::quad_prim<T>, per_face_binding>
{
    enum { value = 1 };
};
template <typename T>
struct num_normals<QUAD_NS::quad_prim<T>, per_vertex_binding>
{
    enum { value = 4 };
};
template <typename T>
struct num_tex_coords<QUAD_NS::quad_prim<T>>
{
    enum { value = 4 };
};

} // visionaray
