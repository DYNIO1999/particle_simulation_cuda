#include "test.h"
#include <cmath>
#include <chrono>
#include <omp.h>
#include <algorithm>

void updateParticlesCPUwithMPI(float deltaTime) {

#pragma omp parallel for
    for (int i = 0; i < MAX_PARTICLES; i++) {
        //int numberOfThreads = omp_get_num_threads();
        //printf("Number of threads: %d", numberOfThreads);

        particles[i].position.x += particles[i].speed.x * deltaTime;
        particles[i].position.y += particles[i].speed.y * deltaTime;

        particles[i].position.x = std::clamp(particles[i].position.x, 0.0f, (float)(SCREEN_WIDTH));
        particles[i].position.y = std::clamp(particles[i].position.y, 0.0f, (float)(SCREEN_HEIGHT));
        float angleChange = 1.0f;
        float angle = std::atan2(particles[i].speed.y, particles[i].speed.x) + angleChange;
        particles[i].speed.x = std::cos(angle);
        particles[i].speed.y = std::sin(angle);

        particles[i].speed.x += 1000.0f * std::cos(angle) * deltaTime;
        particles[i].speed.y += 1000.0f * std::sin(angle) * deltaTime;
    }
}

void updateParticlesCPU(float deltaTime) {

    for (int i = 0; i < MAX_PARTICLES; i++) {
        //int numberOfThreads = omp_get_num_threads();
        //printf("Number of threads: %d", numberOfThreads);

        particles[i].position.x += particles[i].speed.x * deltaTime;
        particles[i].position.y += particles[i].speed.y * deltaTime;

        particles[i].position.x = std::clamp(particles[i].position.x, 0.0f, (float)(SCREEN_WIDTH));
        particles[i].position.y = std::clamp(particles[i].position.y, 0.0f, (float)(SCREEN_HEIGHT));
        float angleChange = 1.0f;
        float angle = std::atan2(particles[i].speed.y, particles[i].speed.x) + angleChange;
        particles[i].speed.x = std::cos(angle);
        particles[i].speed.y = std::sin(angle);

        particles[i].speed.x += 1000.0f * std::cos(angle) * deltaTime;
        particles[i].speed.y += 1000.0f * std::sin(angle) * deltaTime;
    }
}


int main() {

    float res = (1.00467 - 0.756512)/2.0;
    std::cout<<"RESULT: "<<res<<'\n';

    bool isParticleGpuOn = false;
    bool isParticleCPUOn = true;

    InitWindow(SCREEN_WIDTH, SCREEN_HEIGHT,"Particle Simulation");

    for (int i = 0; i < MAX_PARTICLES; i++) {
        particles[i].position = (Vector2){ float(GetRandomValue(0, SCREEN_WIDTH)), float(GetRandomValue(0, SCREEN_HEIGHT)) };
        particles[i].speed = (Vector2){ float(GetRandomValue(-MAX_SPEED, MAX_SPEED)), float(GetRandomValue(-MAX_SPEED, MAX_SPEED))};
        particles[i].color = (Color){
                (unsigned char)GetRandomValue(50, 255),
                (unsigned char)GetRandomValue(50, 255),
                (unsigned char)GetRandomValue(50, 255),
                255
        };

    }

    while (!WindowShouldClose()) {

        float deltatime  = GetFrameTime();

        BeginDrawing();

        ClearBackground(RAYWHITE);


        if(IsKeyPressed(KEY_G)){
            isParticleGpuOn = true;
            isParticleCPUOn = false;
        }
        if(IsKeyPressed(KEY_C)) {
            isParticleGpuOn = false;
            isParticleCPUOn = true;
        }


        if(isParticleCPUOn){
            std::chrono::high_resolution_clock::time_point start_time, end_time;

            start_time = std::chrono::high_resolution_clock::now();

            updateParticlesCPU(deltatime);

            end_time = std::chrono::high_resolution_clock::now();

            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

            std::cout << "Elapsed Time on CPU: " << duration.count() << " us" << std::endl;
        }


        if(isParticleGpuOn){
            std::chrono::high_resolution_clock::time_point start_time, end_time;
            start_time = std::chrono::high_resolution_clock::now();
            updateParticlesCPUwithMPI(deltatime);
            end_time = std::chrono::high_resolution_clock::now();

            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

            std::cout << "Elapsed Time on CPU using OpenMP: " << duration.count() << " us" << std::endl;
        }

        std::chrono::high_resolution_clock::time_point start_time, end_time;

        start_time = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < MAX_PARTICLES; i++) {

            DrawCircle(particles[i].position.x, particles[i].position.y, 3, particles[i].color);
        }

        end_time = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        std::cout << "Rendering Time: " << duration.count() << " ms" << std::endl;


        EndDrawing();
    }

    CloseWindow();

    return 0;
}