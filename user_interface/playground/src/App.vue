<script setup lang="ts">
import Player from "./Player.vue";

const videoUrls = [
    'rtsp://127.0.0.1:8554/input',
];
</script>

<template>
  <div class="playground">
    <template v-for="(url, index) in videoUrls" :key="index">
      <Player :player-key="`${index}`" :video-url="url" />
    </template>
  </div>
</template>

<style scoped>
.playground {
  display: grid;
  grid-template-columns: 1fr 1fr;
  grid-template-rows: 1fr 1fr;
  grid-gap: 2rem;
}
</style>
