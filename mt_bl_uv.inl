namespace QUAD_NS
{

namespace detail {

template <typename T, typename U>
VSNRAY_FUNC
inline hit_record<basic_ray<T>, primitive<unsigned>> intersect_mt_bl_uv(
        basic_ray<T> const&                     ray,
        basic_quad<U> const&                    quad
        )
{

    typedef vector<3, T> vec_type;

    hit_record<basic_ray<T>, primitive<unsigned>> result;
    result.t = T(-1.0);

    // load first vertex only once
    vector<3, U> quad_v1(quad.v1);

    // case T != U
    vec_type v1(quad_v1);
    vec_type e1(quad.v2 - quad_v1);
    vec_type e2(quad.v4 - quad_v1);

    // actual intersection test
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

    result.hit &= ( b1 >= T(0.0));

    if ( !any(result.hit) )
    {
        return result;
    }

    vec_type s2 = cross(d, e1);
    T b2 = dot(ray.dir, s2) * inv_div;

    result.hit &= ( b2 >= T(0.0));

    if ( !any(result.hit) )
    {
        return result;
    }

    // calculate (u,v)-coordinates of the 4th quad-vector relative to the triangle (v1, e1, e2)

    vector<3, U> quad_e1 = quad.v2 - quad_v1;
    vector<3, U> quad_e2 = quad.v4 - quad_v1;

    vector<3, U> quad_d = quad.v3 - quad_v1;
    vector<3, U> quad_s2 = cross(quad_d, quad_e1); // this is a normal of the triangle, use as ray dir

    vector<3, U> quad_s1 = cross(quad_s2, quad_e2);
    U quad_div = dot(quad_s1, quad_e1);
    U quad_inv_div = U(1.0) / quad_div;

    U quad_v4_x = dot(quad_d, quad_s1) * quad_inv_div;
    U quad_v4_y = dot(quad_s2, quad_s2) * quad_inv_div;

    // convert to simd type
    T v4_x = T(dot(quad_d, quad_s1) * quad_inv_div);
    T v4_y = T(dot(quad_s2, quad_s2) * quad_inv_div);

    // (b1,b2) are (u,v)-coordinates (relative to the triangle (v1,e1,e2)) of
    // the intersection point - check if they lie inside quad

    result.hit &= ( b2 <= ((v4_y-T(1.0)) / v4_x) * b1 + T(1.0) );
    result.hit &= ( b1 <= ((v4_x-T(1.0)) / v4_y) * b2 + T(1.0) );

    if ( !any(result.hit) )
    {
        return result;
    }

#if CALCULATE_UV
    // now calculate bilinear coordinates relative to the quad
    T u, v;

    // special cases:
    if (quad_v4_x == 1)
    {
        u = b1;
        v = b2 / (u * (v4_y - T(1.0)) + T(1.0));
    }
    else if (quad_v4_y == 1)
    {
        v = b2;
        u = b1 / (v * (v4_x - T(1.0)) + T(1.0));
    }
    else
    {
        // solve A*u^2 + B*u + C = 0
        T A = -(v4_y - T(1.0));
        T B = b1 * (v4_y - T(1.0)) - b2 * (v4_x - T(1.0)) - T(1.0);
        T C = b1;

        T D = B * B - T(4.0) * A * C;
        T Q = -T(0.5) * (B + copysign(sqrt(D), B));

        u = Q / A;
        u = select(u < T(0.0) || u > T(1.0), C / Q, u);
        v = b2 / (u * (v4_y - T(1.0)) + T(1.0));
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

} // QUAD_NS
