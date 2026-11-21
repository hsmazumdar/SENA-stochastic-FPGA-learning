module class5_tree18(input wire [467:0] i, output wire [0:0] o);

reg [0:0] o_reg;
assign o = o_reg;

always @(*) begin
	if(((~i[8])) | ((~i[239]) & (~i[331]) & i[9] & i[8]) | (i[156] & i[437] & i[331] & i[9] & i[8]) | ((~i[148]) & (~i[437]) & i[331] & i[9] & i[8]) | ((~i[46]) & i[239] & (~i[331]) & i[9] & i[8]))
		o_reg = 0;
	else
		o_reg = 1;
end

endmodule
