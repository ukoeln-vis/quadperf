#pragma once

#include <visionaray/math/math.h>

#include <visionaray/tags.h>
#include <visionaray/get_normal.h>
#include <visionaray/get_shading_normal.h>
#include <visionaray/get_tex_coord.h>
#include <visionaray/intersector.h>

#ifndef CALCULATE_UV
#define CALCULATE_UV 1
#endif

#include "util.inl"

namespace QUAD_NS
{

using namespace visionaray;

//-------------------------------------------------------------------------------------------------
// Quad primitive
//

template <typename T>
struct basic_quad //: primitive<unsigned>
{
    vector<3, T> v1;
    vector<3, T> v2;
    vector<3, T> v3;
    vector<3, T> v4;


    static basic_quad<float> make_quad(
                vector<3, float> const& v1,
                vector<3, float> const& v2,
                vector<3, float> const& v3,
                vector<3, float> const& v4
            )
    {
        basic_quad<float> t;
        t.v1 = v1;
        t.v2 = v2;
        t.v3 = v3;
        t.v4 = v4;
        return t;
    }
};


namespace detail
{

//-------------------------------------------------------------------------------------------------
// Get uv
//

template <typename T>
VSNRAY_FUNC
vector<2, T> get_uv(basic_quad<float> const& quad, vector<3, T> const& isect_pos)
{
    // Glassner 1989, "An Introduction to Ray Tracing", p.60

    using V = vector<3, T>;

    vector<2, T> uv;

    // possibly precalculate ------------------------------

    V e1(quad.v2 - quad.v1);
    V e2(quad.v3 - quad.v2);

    V P_n = cross(e1, e2);
    V P_a(quad.v1 - quad.v2 + quad.v3 - quad.v4);
    V P_b(quad.v2 - quad.v1);
    V P_c(quad.v4 - quad.v1);
    V P_d(quad.v1);

    V N_a = cross(P_a, P_n);
    V N_b = cross(P_b, P_n);
    V N_c = cross(P_c, P_n);

    T D_u0 = dot(N_c, P_d);
    T D_u1 = dot(N_a, P_d) + dot(N_c, P_b);
    T D_u2 = dot(N_a, P_b);

    T D_v0 = dot(N_b, P_d);
    T D_v1 = dot(N_a, P_d) + dot(N_b, P_c);
    T D_v2 = dot(N_a, P_c);

    //-----------------------------------------------------


    // Compute the distance to the plane perpendicular to
    // the quad's "u-axis"...
    //
    // D(u) = (N_c + N_a * u) . (P_d + P_b * u)
    //
    // ... with regards to isect_pos:

    V R_i = isect_pos;

    //
    // D_r(u) = (N_c + N_a * u) . R_i
    //
    // by letting D(u) = D_r(u) and solving the corresponding
    // quadratic equation.

    V Q_ux = N_a / (T(2.0) * D_u2);
    T D_ux = -D_u1 / (T(2.0) * D_u2);
    V Q_uy = -N_c / D_u2;
    T D_uy = D_u0 / D_u2;


    T K_a = D_ux + dot(Q_ux, R_i);
    T K_b = D_uy + dot(Q_uy, R_i);

    //auto parallel_u = (abs(D_u2) < T(0.000001));
    auto parallel_u = (D_u2 == T(0.0));
    uv.x = select(
            parallel_u,
            (dot(N_c, R_i) - D_u0) / (D_u1 - dot(N_a, R_i)),
            K_a - sqrt(K_a * K_a - K_b)
            );

    uv.x = select(
            !parallel_u && (uv.x < T(0.0) || uv.x > T(1.0)),
            K_a + sqrt(K_a * K_a - K_b),
            uv.x
            );


    // Do the same for v

    V Q_vx = N_a / (T(2.0) * D_v2);
    T D_vx = -D_v1 / (T(2.0) * D_v2);
    V Q_vy = -N_b / D_v2;
    T D_vy = D_v0 / D_v2;


    K_a = D_vx + dot(Q_vx, R_i);
    K_b = D_vy + dot(Q_vy, R_i);


    //auto parallel_v = (abs(D_v2) < T(0.0001));
    auto parallel_v = (D_v2 == T(0.0));
    uv.y = select(
            parallel_v,
            (dot(N_b, R_i) - D_v0) / (D_v1 - dot(N_a, R_i)),
            K_a - sqrt(K_a * K_a - K_b)
            );

    uv.y = select(
            !parallel_v && (uv.y < T(0.0) || uv.y > T(1.0)),
            K_a + sqrt(K_a * K_a - K_b),
            uv.y
            );


    return uv;
}

} // detail

} // QUAD_NS


#include "pluecker.inl"
#include "project2d.inl"
#include "uv.inl"
#include "mt_bl_uv.inl"


namespace QUAD_NS
{

//-------------------------------------------------------------------------------------------------
// Interface
//

template <typename R>
VSNRAY_FUNC
hit_record<R, primitive<unsigned>> intersect(R const& ray, basic_quad<float> const& quad)
{
    return detail::intersect_mt_bl_uv(ray, quad);
    // return detail::intersect_pluecker(ray, quad);
    // return detail::intersect_project_2D(ray, quad);
    // return detail::intersect_uv(ray, quad);
}

struct quad_intersector_mt_bl_uv : basic_intersector<quad_intersector_mt_bl_uv>
{
    using basic_intersector<quad_intersector_mt_bl_uv>::operator();

    template <typename R, typename S>
    VSNRAY_FUNC
    auto operator()(
            R const& ray,
            basic_quad<S> const& quad
            )
        -> decltype( detail::intersect_mt_bl_uv(ray, quad) )
    {
        return detail::intersect_mt_bl_uv(ray, quad);
    }
};

struct quad_intersector_pluecker : basic_intersector<quad_intersector_pluecker>
{
    using basic_intersector<quad_intersector_pluecker>::operator();

    template <typename R, typename S>
    VSNRAY_FUNC
    auto operator()(
            R const& ray,
            basic_quad<S> const& quad
            )
        -> decltype( detail::intersect_pluecker(ray, quad) )
    {
        return detail::intersect_pluecker(ray, quad);
    }
};

struct quad_intersector_project_2D : basic_intersector<quad_intersector_project_2D>
{
    using basic_intersector<quad_intersector_project_2D>::operator();

    template <typename R, typename S>
    VSNRAY_FUNC
    auto operator()(
            R const& ray,
            basic_quad<S> const& quad
            )
        -> decltype( detail::intersect_project_2D(ray, quad) )
    {
        return detail::intersect_project_2D(ray, quad);
    }
};

struct quad_intersector_uv : basic_intersector<quad_intersector_uv>
{
    using basic_intersector<quad_intersector_uv>::operator();

    template <typename R, typename S>
    VSNRAY_FUNC
    auto operator()(
            R const& ray,
            basic_quad<S> const& quad
            )
        -> decltype( detail::intersect_uv(ray, quad) )
    {
        return detail::intersect_uv(ray, quad);
    }
};



template <typename T>
VSNRAY_CPU_FUNC
basic_aabb<T> get_bounds(basic_quad<T> const& t)
{
    basic_aabb<T> bounds;

    bounds.invalidate();
    bounds.insert(t.v1);
    bounds.insert(t.v2);
    bounds.insert(t.v3);
    bounds.insert(t.v4);

    return bounds;
}

template <typename T>
VSNRAY_CPU_FUNC
void split_primitive(aabb& L, aabb& R, float plane, int axis, QUAD_NS::basic_quad<T> const& prim)
{
    L.invalidate();
    R.invalidate();

    visionaray::detail::split_edge(L, R, prim.v1, prim.v2, plane, axis);
    visionaray::detail::split_edge(L, R, prim.v2, prim.v3, plane, axis);
    visionaray::detail::split_edge(L, R, prim.v3, prim.v4, plane, axis);
    visionaray::detail::split_edge(L, R, prim.v4, prim.v1, plane, axis);
}


template <typename HR, typename T>
VSNRAY_FUNC
inline vector<3, T> get_normal(HR const hr, basic_quad<T> const& quad)
{
    VSNRAY_UNUSED(hr);
    return normalize( cross(quad.v2 - quad.v1, quad.v3 - quad.v1) );
}

template <typename Normals, typename HR, typename T>
VSNRAY_FUNC
inline auto get_normal(
        Normals                     normals,
        HR const&                   hr,
        QUAD_NS::basic_quad<T>      prim,
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
        QUAD_NS::basic_quad<T>      /* */,
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


template <typename TexCoords, typename R, typename T>
VSNRAY_FUNC
inline auto get_tex_coord(
        TexCoords                                   tex_coords,
        hit_record<R, primitive<unsigned>> const&   hr,
        basic_quad<T>                               /* */
        )
    -> typename std::iterator_traits<TexCoords>::value_type
{
    auto t1 = tex_coords[hr.prim_id * 4];
    auto t2 = tex_coords[hr.prim_id * 4 + 1];
    auto t3 = tex_coords[hr.prim_id * 4 + 2];
    auto t4 = tex_coords[hr.prim_id * 4 + 3];

    auto t11 = lerp(t1, t2, hr.u);
    auto t12 = lerp(t3, t4, hr.u);

    return lerp(t11, t12, hr.v);
}

} // QUAD_NS


namespace visionaray
{

template <typename T>
struct num_vertices<QUAD_NS::basic_quad<T>>
{
    enum { value = 4 };
};
template <typename T>
struct num_normals<QUAD_NS::basic_quad<T>, per_face_binding>
{
    enum { value = 1 };
};
template <typename T>
struct num_normals<QUAD_NS::basic_quad<T>, per_vertex_binding>
{
    enum { value = 4 };
};
template <typename T>
struct num_tex_coords<QUAD_NS::basic_quad<T>>
{
    enum { value = 4 };
};

} // visionaray
