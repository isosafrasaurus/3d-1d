#include <iostream>
#include <vector>
#include <cmath>
#include <limits>

struct Vec3 {
    float x, y, z;
    Vec3() : x(0), y(0), z(0) {}
    Vec3(float _x, float _y, float _z) : x(_x), y(_y), z(_z) {}
};

Vec3 operator-(const Vec3 &a, const Vec3 &b) {
    return Vec3(a.x - b.x, a.y - b.y, a.z - b.z);
}

Vec3 operator+(const Vec3 &a, const Vec3 &b) {
    return Vec3(a.x + b.x, a.y + b.y, a.z + b.z);
}

Vec3 operator*(const Vec3 &a, float s) {
    return Vec3(a.x * s, a.y * s, a.z * s);
}

float dot(const Vec3 &a, const Vec3 &b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

float length(const Vec3 &a) {
    return std::sqrt(dot(a, a));
}

float clamp(float v, float lo, float hi) {
    if(v < lo) return lo;
    if(v > hi) return hi;
    return v;
}

float pointSegmentDistance(const Vec3 &p, const Vec3 &a, const Vec3 &b) {
    Vec3 ab = b - a;
    float abLenSq = dot(ab, ab);
    if(abLenSq == 0.0f) return length(p - a);
    float t = clamp(dot(p - a, ab) / abLenSq, 0.0f, 1.0f);
    Vec3 proj = a + ab * t;
    return length(p - proj);
}

struct KDNode {
    Vec3 pos;
    KDNode *left;
    KDNode *right;
    KDNode *parent;
    float edgeRadius;
    KDNode(const Vec3 &p, float rad, KDNode *par)
        : pos(p), left(nullptr), right(nullptr), parent(par), edgeRadius(rad) {}
};

class KDTree {
public:
    KDTree() : root(nullptr) {}
    ~KDTree() { clear(root); }
    void insert(const Vec3 &p, float radius) {
        if(root == nullptr) {
            root = new KDNode(p, 0.0f, nullptr);
        } else {
            insertRec(root, nullptr, p, radius, 0);
        }
    }
    float query(const Vec3 &p) {
        return queryRec(root, p);
    }
private:
    KDNode *root;
    void clear(KDNode *node) {
        if(node) {
            clear(node->left);
            clear(node->right);
            delete node;
        }
    }
    void insertRec(KDNode *node, KDNode *parent, const Vec3 &p, float radius, int depth) {
        int axis = depth % 3;
        float nodeVal = (axis == 0 ? node->pos.x : (axis == 1 ? node->pos.y : node->pos.z));
        float pVal = (axis == 0 ? p.x : (axis == 1 ? p.y : p.z));
        if(pVal < nodeVal) {
            if(node->left == nullptr) {
                node->left = new KDNode(p, radius, node);
            } else {
                insertRec(node->left, node, p, radius, depth + 1);
            }
        } else {
            if(node->right == nullptr) {
                node->right = new KDNode(p, radius, node);
            } else {
                insertRec(node->right, node, p, radius, depth + 1);
            }
        }
    }
    float queryRec(KDNode *node, const Vec3 &p) {
        if(node == nullptr) return 0.0f;
        float ret = 0.0f;
        if(node->parent != nullptr) {
            float d = pointSegmentDistance(p, node->parent->pos, node->pos);
            if(d <= node->edgeRadius) return node->edgeRadius;
        }
        ret = queryRec(node->left, p);
        if(ret > 0.0f) return ret;
        ret = queryRec(node->right, p);
        return ret;
    }
};