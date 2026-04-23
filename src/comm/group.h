#pragma once

#include "comm/node.h"

#include <vector>

namespace ooverlap {
namespace comm {

struct Group {
    std::vector<int> devices{};
    std::vector<Node> nodes{};
};

void group_init(
    Group* group,
    const std::vector<int>& devices);

void group_destroy(
    Group* group);

Node* group_get_node(
    Group* group,
    int rank);

const Node* group_get_node(
    const Group* group,
    int rank);

} // namespace comm
} // namespace ooverlap
