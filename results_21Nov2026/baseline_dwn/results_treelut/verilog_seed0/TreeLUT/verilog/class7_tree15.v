module class7_tree15(input wire [467:0] i, output wire [0:0] o);

reg [0:0] o_reg;
assign o = o_reg;

always @(*) begin
	if(((~i[24])) | ((~i[251]) & (~i[177]) & i[24]) | (i[152] & (~i[215]) & i[208] & i[177] & i[24]) | (i[293] & i[147] & (~i[208]) & i[177] & i[24]) | ((~i[64]) & (~i[147]) & (~i[208]) & i[177] & i[24]))
		o_reg = 0;
	else
		o_reg = 1;
end

endmodule
