package com.rigil.particleeffect;

import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.ScrollView;
import android.widget.TextView;

import com.plattysoft.leonids.ParticleSystem;

public class MainActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        Button button = findViewById(R.id.button1);

        final ParticleSystem ps = new ParticleSystem(this, 1000, R.drawable.ic_launcher_background, 100)
                .setSpeedRange(0.2f, 0.5f);
//                .oneShot(button, 500);

        button.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                ps.oneShot(view, 500);
            }
        });
    }

    @Override
    protected void onResume() {
        super.onResume();

        ScrollView scroll = findViewById(R.id.scroll);
//        ParticleSystem ps = new ParticleSystem(this, 1000, R.drawable.ic_launcher_background, 100)
//                .setSpeedRange(0.2f, 0.5f)
//                .oneShot(scroll, 500);
        new ParticleSystem(this, 1000, R.drawable.ic_launcher_background, 100)
                .setSpeedRange(0.2f, 0.5f)
                .oneShot(scroll, 500);

        new ParticleSystem(this, 80, R.drawable.star_white, 10000)
                .setSpeedModuleAndAngleRange(0f, 0.3f, 180, 180)
                .setRotationSpeed(144)
                .setAcceleration(0.00005f, 90)
                .emit(findViewById(R.id.emiter_top_right), 8);

        new ParticleSystem(this, 80, R.drawable.star_white, 10000)
                .setSpeedModuleAndAngleRange(0f, 0.3f, 0, 0)
                .setRotationSpeed(144)
                .setAcceleration(0.00005f, 90)
                .emit(findViewById(R.id.emiter_top_left), 8);

    }

    public void onClick() {
        Button button = findViewById(R.id.button1);

        new ParticleSystem(this, 1000, R.drawable.ic_launcher_background, 100)
                .setSpeedRange(0.2f, 0.5f)
                .oneShot(button, 500);
    }

    private void sprayParticle() {
        Button textView = findViewById(R.id.button1);

        new ParticleSystem(this, 1000, R.drawable.ic_launcher_background, 100)
                .setSpeedRange(0.2f, 0.5f)
                .oneShot(textView, 500);
    }
}
