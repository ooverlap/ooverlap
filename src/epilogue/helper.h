#pragma once

#include <type_traits>

// --------------------------------------------------------------------------
// CUTLASS 3.9.0: try several places for NumEpilogueWarpGroups
// --------------------------------------------------------------------------
template <class T, class = void>
struct EpiWGsFromBase {
  static constexpr int value = 0;
};

template <class T>
struct EpiWGsFromBase<T, std::void_t<decltype(T::NumEpilogueWarpGroups)>> {
  static constexpr int value = int(T::NumEpilogueWarpGroups);
};

template <class T, class = void>
struct EpiWGsFromDispatchPolicy {
  static constexpr int value = 0;
};

template <class T>
struct EpiWGsFromDispatchPolicy<T, std::void_t<decltype(T::DispatchPolicy::NumEpilogueWarpGroups)>> {
  static constexpr int value = int(T::DispatchPolicy::NumEpilogueWarpGroups);
};

template <class T, class = void>
struct EpiWGsFromSchedule {
  static constexpr int value = 0;
};

template <class T>
struct EpiWGsFromSchedule<T, std::void_t<decltype(T::DispatchPolicy::Schedule::NumEpilogueWarpGroups)>> {
  static constexpr int value = int(T::DispatchPolicy::Schedule::NumEpilogueWarpGroups);
};
